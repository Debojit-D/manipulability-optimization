#!/usr/bin/env python3
# nullspace_opt.py

from __future__ import annotations
import numpy as np
import mujoco

class NullSpaceOptimizer:
    """
    Reusable null-space controller with manipulability optimization for MuJoCo arms.

    Features
    --------
    - Task: 'pose6' (hold position+orientation) or 'pos3' (hold position only)
    - Null motion policies:
        'svd'     : last right-singular vector of J (classic 6x7 → 1D null)
        'random'  : projected random direction in null space
        'manigrad': projected gradient ascent on manipulability (default)
    - Limit aware: soft margins + posture bias (both applied in null space)
    - Clean API: set_target, set_task_mode, step()

    Typical loop
    ------------
        opt = NullSpaceOptimizer(model, data, arm_joints, ee_body="hand")
        opt.set_target_from_current()        # lock current pose as the task
        while running:
            opt.step()
    """

    # -------------------------- ctor & config --------------------------
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joints: list[str],
        ee_body: str,
        *,
        dt: float = 1.0/300.0,
        task_mode: str = "pose6",      # 'pose6' | 'pos3'
        k_pos: float = 0.01,
        k_ori: float = 0.01,
        damp_lambda: float = 1e-2,
        alpha_null: float = 0.12,
        qdot_max: float = 1.0,
        soft_margin_frac: float = 0.10,
        k_posture: float = 0.5,
        null_policy: str = "manigrad",  # 'manigrad' | 'svd' | 'random'
        gripper_ctrl_index: int | None = 7,   # None if no gripper actuator
        gripper_open_cmd: float = 255.0,
        fd_eps: float = 1e-4,           # finite-diff step for mani gradient
        seed: int = 42,
    ):
        self.m = model
        self.d = data
        self.dt = dt

        # task / control params
        self.task_mode = task_mode.lower()
        assert self.task_mode in ("pose6", "pos3")
        self.k_pos = float(k_pos)
        self.k_ori = float(k_ori)
        self.lmbd = float(damp_lambda)
        self.alpha = float(alpha_null)
        self.qdot_max = float(qdot_max)
        self.null_policy = null_policy.lower()
        assert self.null_policy in ("manigrad", "svd", "random")

        # limit handling
        self.soft_margin_frac = float(soft_margin_frac)
        self.k_posture = float(k_posture)

        # manipulability grad
        self.eps = float(fd_eps)

        # joints & kinematics indexing
        self.joint_ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_joints]
        self.qaddr = np.array([self.m.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.daddr = np.array([self.m.jnt_dofadr[jid]  for jid in self.joint_ids], dtype=int)

        # ee body id
        self.ee_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, ee_body)

        # target (filled later)
        self.p_target = None
        self.q_target = None  # wxyz

        # internal reference for position control (what actuators will track)
        self.q_ref = self.d.qpos[self.qaddr].copy()

        # joint limits
        self.qmin = np.array([self.m.jnt_range[jid][0] for jid in self.joint_ids])
        self.qmax = np.array([self.m.jnt_range[jid][1] for jid in self.joint_ids])
        self.qmid = 0.5 * (self.qmin + self.qmax)
        self.span = (self.qmax - self.qmin).clip(min=1e-9)

        # jacobian work buffers
        self._jacp = np.zeros((3, self.m.nv))
        self._jacr = np.zeros((3, self.m.nv))

        # gripper
        self.gripper_ctrl_index = gripper_ctrl_index
        self.gripper_open_cmd = float(gripper_open_cmd)

        # rng
        self.rng = np.random.default_rng(seed)

        # small caches for overlays/debug
        self.last_mani = 0.0
        self.last_pos_err = 0.0
        self.last_ori_err = 0.0
        self.last_nullity = 0

    # -------------------------- public API --------------------------
    def set_target_from_current(self):
        mujoco.mj_forward(self.m, self.d)
        self.p_target = self.d.xpos[self.ee_bid].copy()
        self.q_target = self.d.xquat[self.ee_bid].copy()

    def set_task_mode(self, mode: str):
        mode = mode.lower()
        assert mode in ("pose6", "pos3")
        self.task_mode = mode

    def set_null_policy(self, policy: str):
        policy = policy.lower()
        assert policy in ("manigrad", "svd", "random")
        self.null_policy = policy

    def step(self):
        """Compute one control step and write desired joint positions to data.ctrl."""
        mujoco.mj_forward(self.m, self.d)

        # 1) current EE pose
        p = self.d.xpos[self.ee_bid].copy()
        q = self.d.xquat[self.ee_bid].copy()

        # 2) Jacobian for our DOFs
        self._jacp.fill(0.0); self._jacr.fill(0.0)
        mujoco.mj_jacBody(self.m, self.d, self._jacp, self._jacr, self.ee_bid)
        Jp = self._jacp[:, self.daddr]  # 3x7
        Jr = self._jacr[:, self.daddr]  # 3x7

        # 3) task vector & Jacobian
        e_pos = (self.p_target - p)
        if self.task_mode == "pos3":
            e_task = self.k_pos * e_pos               # 3,
            J = Jp                                    # 3x7
        else:
            q_err = self._quat_mul(self._quat_conj(q), self.q_target)
            e_ori = self._quat_to_rotvec(q_err)
            e_task = np.hstack([self.k_pos*e_pos, self.k_ori*e_ori])  # 6,
            J = np.vstack([Jp, Jr])                                    # 6x7

        # 4) task joint velocity (damped least squares)
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + (self.lmbd**2) * np.eye(J.shape[0]))
        qdot_task = J_pinv @ e_task

        # 5) null projector
        N = np.eye(7) - J_pinv @ J

        # 6) choose a null-space direction
        if self.null_policy == "manigrad":
            g = self._manipulability_grad(J)         # ∂w/∂q (7,)
            z = N @ g                                 # project into null space
            if np.linalg.norm(z) < 1e-12:
                z = self._svd_null(J)                 # fallback
            z = z / (np.linalg.norm(z) + 1e-12)
        elif self.null_policy == "svd":
            z = self._svd_null(J)
        else:  # random
            z = N @ self.rng.standard_normal(7)
            z = z / (np.linalg.norm(z) + 1e-12)

        # 7) limit-aware shaping (weights + posture pull) INSIDE null space
        q_now = self.d.qpos[self.qaddr].copy()
        w_lim = self._limit_weights(q_now)
        z = w_lim * z  # fade near limits
        z += self.k_posture * (self.qmid - q_now) / self.span  # posture bias
        z = N @ z      # ensure we stay in null space after adding posture
        z = z / (np.linalg.norm(z) + 1e-12)

        qdot = qdot_task + self.alpha * z

        # 8) near-limit guard + cap + integrate + clamp
        margin = self.soft_margin_frac * self.span
        near_hi = (q_now > (self.qmax - margin)) & (qdot > 0)
        near_lo = (q_now < (self.qmin + margin)) & (qdot < 0)
        qdot[near_hi | near_lo] = 0.0

        qdot = np.clip(qdot, -self.qdot_max, self.qdot_max)
        self.q_ref = self.q_ref + self.dt * qdot
        self.q_ref = self._clamp_to_limits(self.q_ref)

        # 9) write to actuators (assumes first 7 actuators map 1:1 to joints)
        self.d.ctrl[0:7] = self.q_ref
        if self.gripper_ctrl_index is not None and self.gripper_ctrl_index < self.m.nu:
            self.d.ctrl[self.gripper_ctrl_index] = self.gripper_open_cmd

        # 10) diagnostics
        self.last_mani = self._manipulability(J)
        self.last_pos_err = float(np.linalg.norm(e_pos))
        if self.task_mode == "pose6":
            self.last_ori_err = float(np.linalg.norm(self._quat_to_rotvec(self._quat_mul(self._quat_conj(q), self.q_target))))
        else:
            self.last_ori_err = 0.0
        self.last_nullity = self._nullity(J)

        # step sim one step if you want physics to evolve here (optional)
        mujoco.mj_step(self.m, self.d)

    # -------------------------- utilities: math/grad --------------------------
    def _clamp_to_limits(self, q):
        out = q.copy()
        for i, jid in enumerate(self.joint_ids):
            lo, hi = self.m.jnt_range[jid]
            lo2, hi2 = (min(lo, hi), max(lo, hi))
            out[i] = np.clip(out[i], lo2, hi2)
        return out

    def _limit_weights(self, q):
        span = self.span
        margin = self.soft_margin_frac * span
        lo = self.qmin + margin
        hi = self.qmax - margin
        d = np.minimum(q - lo, hi - q)  # distance to closest soft boundary
        w = np.clip(d / (margin + 1e-12), 0.0, 1.0) ** 2
        return w

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
    def _quat_conj(q):
        w,x,y,z = q
        return np.array([w,-x,-y,-z])

    @staticmethod
    def _quat_to_rotvec(q):
        q = q / (np.linalg.norm(q) + 1e-12)
        w, v = q[0], q[1:]
        nv = np.linalg.norm(v)
        w = np.clip(w, -1.0, 1.0)
        if nv < 1e-12:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(nv, w)
        if angle < 1e-12:
            return np.zeros(3)
        axis = v / (nv + 1e-12)
        return angle * axis

    @staticmethod
    def _nullity(J):
        # robust rank with threshold
        s = np.linalg.svd(J, compute_uv=False)
        rank = (s > 1e-6).sum()
        return J.shape[1] - rank

    @staticmethod
    def _manipulability(J):
        # sqrt(det(J J^T)) with tiny Tikhonov for safety
        JJt = J @ J.T
        JJt += 1e-12 * np.eye(JJt.shape[0])
        return float(np.sqrt(np.linalg.det(JJt)))

    def _svd_null(self, J):
        U, S, Vt = np.linalg.svd(J, full_matrices=True)
        v = Vt.T[:, -1]
        return v / (np.linalg.norm(v) + 1e-12)

    def _manipulability_grad(self, J):
        """
        Finite-difference gradient of manipulability w.r.t. the 7 joints.
        Uses in-place perturbation of qpos; restores state afterward.
        Gradient is evaluated at current configuration.
        """
        base_q = self.d.qpos[self.qaddr].copy()
        base = self._manipulability(J)
        g = np.zeros(7)

        for i in range(7):
            # forward diff
            self.d.qpos[self.qaddr] = base_q
            self.d.qpos[self.qaddr][i] += self.eps
            mujoco.mj_forward(self.m, self.d)
            Jp, Jr = self._current_Jp_Jr()
            J_task = Jp if self.task_mode == "pos3" else np.vstack([Jp, Jr])
            f = self._manipulability(J_task)
            g[i] = (f - base) / self.eps

        # restore
        self.d.qpos[self.qaddr] = base_q
        mujoco.mj_forward(self.m, self.d)
        return g

    def _current_Jp_Jr(self):
        """Helper: recompute Jp, Jr at current state for our 7 DOFs."""
        self._jacp.fill(0.0); self._jacr.fill(0.0)
        mujoco.mj_jacBody(self.m, self.d, self._jacp, self._jacr, self.ee_bid)
        Jp = self._jacp[:, self.daddr]
        Jr = self._jacr[:, self.daddr]
        return Jp, Jr
