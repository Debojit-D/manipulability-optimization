#!/usr/bin/env python3
# utils/ik/ik_qp.py

from __future__ import annotations
import numpy as np
import cvxpy as cp
import mujoco

class IKQPController:
    """
    Velocity-level QP IK with null-space optimization (MuJoCo + CVXPY/OSQP).

    Primary objective:
        minimize 0.5*|| J qdot - xdot_des ||^2 + lambda_q*||qdot||^2 + rho_s*||s||^2
        - beta * c^T qdot

    Constraints:
        qmin <= q + dt*qdot <= qmax
        -qdot_max <= qdot <= qdot_max
        (optional) linear null constraints A_null qdot = b_null

    Task modes:
        'pos3'  : hold position only    (3x7 J)
        'pose6' : hold pose (pos+ori)   (6x7 J)

    Null objective 'c':
        - 'manigrad' : projected manipulability gradient (recommended)
        - 'posture'  : posture bias toward mid-range
        - 'none'     : no null preference
        - or pass a callable that returns a 7-vector c(q, J).

    Usage:
        ik = IKQPController(model, data, arm_joints, ee_body="hand")
        ik.set_target_from_current()
        while running:
            qdot = ik.step()         # solves QP, updates data.ctrl (desired q)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joints: list[str],
        ee_body: str,
        *,
        dt: float = 1/300,
        task_mode: str = "pose6",       # 'pos3' | 'pose6'
        k_pos: float = 2.0,             # task gains -> xdot_des = K*e
        k_ori: float = 1.5,
        lambda_q: float = 1e-4,         # Tikhonov on qdot
        rho_s: float = 1e-6,            # slack penalty
        beta: float = 0.2,              # weight for null objective linear term
        qdot_max: float = 1.5,          # |qdot_i| <= qdot_max
        soft_margin_frac: float = 0.05, # shrink joint box a bit
        null_objective: str | callable = "manigrad",  # 'manigrad'|'posture'|'none'|callable
        use_slack: bool = True,
        solver: str = "OSQP",
        gripper_ctrl_index: int | None = 7,
        gripper_open_cmd: float = 255.0,
        fd_eps: float = 1e-4,           # for mani gradient
        seed: int = 0,
    ):
        self.m, self.d = model, data
        self.dt = float(dt)
        self.task_mode = task_mode.lower()
        assert self.task_mode in ("pos3","pose6")
        self.k_pos, self.k_ori = float(k_pos), float(k_ori)
        self.lambda_q, self.rho_s = float(lambda_q), float(rho_s)
        self.beta = float(beta)
        self.qdot_max = float(qdot_max)
        self.soft_margin_frac = float(soft_margin_frac)
        self.null_objective = null_objective
        self.use_slack = use_slack
        self.solver = solver
        self.gripper_ctrl_index = gripper_ctrl_index
        self.gripper_open_cmd = float(gripper_open_cmd)
        self.fd_eps = float(fd_eps)
        self.rng = np.random.default_rng(seed)

        # indexing
        self.joint_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_joints]
        self.qaddr = np.array([self.m.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.daddr = np.array([self.m.jnt_dofadr[jid]  for jid in self.joint_ids], dtype=int)
        self.ee_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, ee_body)

        # limits
        self.qmin = np.array([self.m.jnt_range[jid][0] for jid in self.joint_ids])
        self.qmax = np.array([self.m.jnt_range[jid][1] for jid in self.joint_ids])
        span = (self.qmax - self.qmin).clip(min=1e-9)
        self.qmid = 0.5*(self.qmin + self.qmax)
        self.lo_soft = self.qmin + self.soft_margin_frac*span
        self.hi_soft = self.qmax - self.soft_margin_frac*span

        # target pose (filled by set_target_from_current or set_target)
        self.p_target = None   # 3
        self.q_target = None   # quat (wxyz)

        # desired config for position actuators
        self.q_ref = self.d.qpos[self.qaddr].copy()

        # small jacobian buffers
        self._jacp = np.zeros((3, self.m.nv))
        self._jacr = np.zeros((3, self.m.nv))

        # cvxpy objects (built once, parameters updated every step)
        self._build_qp()

        # diagnostics
        self.last_status = None
        self.last_mani = 0.0
        self.last_err_pos = 0.0
        self.last_err_ori = 0.0

    # ---------------- public API ----------------
    def set_target_from_current(self):
        mujoco.mj_forward(self.m, self.d)
        self.p_target = self.d.xpos[self.ee_bid].copy()
        self.q_target = self.d.xquat[self.ee_bid].copy()

    def set_target(self, p_target: np.ndarray, q_target_wxyz: np.ndarray | None = None):
        self.p_target = np.array(p_target, dtype=float).copy()
        if self.task_mode == "pose6":
            assert q_target_wxyz is not None, "pose6 needs orientation target"
            self.q_target = np.array(q_target_wxyz, dtype=float).copy()

    def step(self):
        """Solve QP for qdot, integrate q_ref, write data.ctrl; return qdot (7,)."""
        mujoco.mj_forward(self.m, self.d)

        # --- kinematics ---
        p = self.d.xpos[self.ee_bid].copy()
        q = self.d.xquat[self.ee_bid].copy()
        self._jacp.fill(0.0); self._jacr.fill(0.0)
        mujoco.mj_jacBody(self.m, self.d, self._jacp, self._jacr, self.ee_bid)
        Jp = self._jacp[:, self.daddr]        # 3x7
        Jr = self._jacr[:, self.daddr]        # 3x7
        if self.task_mode == "pos3":
            J = Jp
            e = (self.p_target - p)
            xdot_des = self.k_pos * e
            self.last_err_pos = float(np.linalg.norm(e))
            self.last_err_ori = 0.0
        else:
            J = np.vstack([Jp, Jr])
            e_pos = (self.p_target - p)
            e_ori = self._quat_log(self._quat_mul(self._quat_conj(q), self.q_target))
            xdot_des = np.hstack([self.k_pos * e_pos, self.k_ori * e_ori])
            self.last_err_pos = float(np.linalg.norm(e_pos))
            self.last_err_ori = float(np.linalg.norm(e_ori))

        # manipulability (for logs)
        self.last_mani = self._manipulability(J)

        # --- null objective vector c ---
        if callable(self.null_objective):
            c = np.asarray(self.null_objective(self, J), dtype=float)
        elif self.null_objective == "manigrad":
            g = self._manipulability_grad(J)          # ∂w/∂q
            # project into true null: N = I - J^+J
            Jpinv = self._damped_pinv(J, 1e-3)
            N = np.eye(7) - Jpinv @ J
            c = N @ g
        elif self.null_objective == "posture":
            q_now = self.d.qpos[self.qaddr].copy()
            c = (self.qmid - q_now) / ((self.hi_soft - self.lo_soft).clip(min=1e-9))
        else:  # 'none'
            c = np.zeros(7)

        # --- update CVXPY params and solve ---
        q_now = self.d.qpos[self.qaddr].copy()
        self._P_J.value = J
        self._p_xdot.value = xdot_des
        self._p_c.value = c

        self._p_q_lo.value = self.lo_soft - q_now
        self._p_q_hi.value = self.hi_soft - q_now
        self._p_qdot_box.value = self.qdot_max * np.ones(7)

        try:
            self.prob.solve(solver=cp.OSQP if self.solver.upper()=="OSQP" else None, warm_start=True, verbose=False)
            status = self.prob.status
        except Exception as e:
            status = f"EXCEPTION: {e}"

        self.last_status = status

        if status not in ("optimal","optimal_inaccurate"):
            # fallback: damped least-squares step
            qdot = self._damped_pinv(J, 1e-2) @ xdot_des
        else:
            qdot = np.asarray(self.v_qdot.value).reshape(-1)

        # integrate and clamp
        self.q_ref = self.q_ref + self.dt * qdot
        self.q_ref = np.clip(self.q_ref, self.qmin, self.qmax)

        # write to MuJoCo position actuators (assuming 1:1 order)
        self.d.ctrl[0:7] = self.q_ref
        if self.gripper_ctrl_index is not None and self.gripper_ctrl_index < self.m.nu:
            self.d.ctrl[self.gripper_ctrl_index] = self.gripper_open_cmd

        # advance sim one step (optional)
        mujoco.mj_step(self.m, self.d)
        return qdot

    # -------------- internal: build QP once --------------
    def _build_qp(self):
        # Variables
        self.v_qdot = cp.Variable(7)                 # decision: joint velocities
        if self.use_slack:
            # s matches xdot dimension (3 or 6) to soften primary task if needed
            self.v_s = cp.Variable(6)                # we will use first 3 for pos3
        else:
            self.v_s = None

        # Params (updated every step)
        self._P_J    = cp.Parameter((6,7))           # will slice to 3x7 for pos3
        self._p_xdot = cp.Parameter(6)
        self._p_c    = cp.Parameter(7)               # null linear term
        self._p_q_lo = cp.Parameter(7)               # box: q_lo <= dt*qdot <= q_hi
        self._p_q_hi = cp.Parameter(7)
        self._p_qdot_box = cp.Parameter(7)

        # Build objective
        if self.task_mode == "pos3":
            Jx = self._P_J[:3, :]
            xdot = self._p_xdot[:3]
        else:
            Jx = self._P_J
            xdot = self._p_xdot

        resid = Jx @ self.v_qdot - xdot
        obj_terms = [0.5 * cp.sum_squares(resid),
                     self.lambda_q * cp.sum_squares(self.v_qdot),
                     - self.beta * self._p_c @ self.v_qdot]
        if self.use_slack:
            # allow tiny relaxation of primary with big penalty
            if self.task_mode == "pos3":
                obj_terms[0] = 0.5 * cp.sum_squares(resid - self.v_s[:3])
            else:
                obj_terms[0] = 0.5 * cp.sum_squares(resid - self.v_s)
            obj_terms.append(self.rho_s * cp.sum_squares(self.v_s))

        objective = cp.Minimize(sum(obj_terms))

        # Constraints
        cons = []
        # joint box over one step
        cons += [ self._p_q_lo <= self.dt * self.v_qdot,
                  self.dt * self.v_qdot <= self._p_q_hi ]
        # velocity box
        cons += [ -self._p_qdot_box <= self.v_qdot,
                   self.v_qdot <= self._p_qdot_box ]

        # Build problem
        self.prob = cp.Problem(objective, cons)

    # -------------- helpers --------------
    @staticmethod
    def _quat_conj(q):
        w,x,y,z = q; return np.array([w,-x,-y,-z])

    @staticmethod
    def _quat_mul(q1, q2):
        w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @staticmethod
    def _quat_log(q):
        """Quaternion log-map -> so(3) vector."""
        q = q / (np.linalg.norm(q) + 1e-12)
        w, v = q[0], q[1:]
        nv = np.linalg.norm(v)
        w = np.clip(w, -1.0, 1.0)
        if nv < 1e-12: return np.zeros(3)
        angle = 2.0*np.arctan2(nv, w)
        if angle < 1e-12: return np.zeros(3)
        axis = v / (nv + 1e-12)
        return angle * axis

    @staticmethod
    def _damped_pinv(J, lam):
        m, n = J.shape
        if m <= n:
            return J.T @ np.linalg.inv(J @ J.T + (lam**2)*np.eye(m))
        else:
            return np.linalg.inv(J.T @ J + (lam**2)*np.eye(n)) @ J.T

    @staticmethod
    def _manipulability(J):
        JJt = J @ J.T
        JJt += 1e-12*np.eye(JJt.shape[0])
        return float(np.sqrt(np.linalg.det(JJt)))

    def _manipulability_grad(self, J):
        """Finite-diff gradient wrt the 7 arm joints at current state."""
        base_q = self.d.qpos[self.qaddr].copy()
        base_w = self._manipulability(J)
        g = np.zeros(7)
        for i in range(7):
            self.d.qpos[self.qaddr] = base_q
            self.d.qpos[self.qaddr][i] += self.fd_eps
            mujoco.mj_forward(self.m, self.d)
            Jp, Jr = self._recompute_J()
            Jt = Jp if self.task_mode == "pos3" else np.vstack([Jp, Jr])
            g[i] = (self._manipulability(Jt) - base_w) / self.fd_eps
        # restore
        self.d.qpos[self.qaddr] = base_q
        mujoco.mj_forward(self.m, self.d)
        return g

    def _recompute_J(self):
        self._jacp.fill(0.0); self._jacr.fill(0.0)
        mujoco.mj_jacBody(self.m, self.d, self._jacp, self._jacr, self.ee_bid)
        return self._jacp[:, self.daddr], self._jacr[:, self.daddr]
