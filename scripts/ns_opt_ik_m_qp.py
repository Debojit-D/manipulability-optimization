#!/usr/bin/env python3
"""
Pose IK with QP (world-frame target) — trajectory to target + hold + null-space manipulability optimization.
- Single stacked LS objective keeps solver structure fixed (prevents OSQP A/P pattern errors).
- Null-space term: maximize manipulability via gradient ascent on log det(J J^T) projected to null-space.
- Convergence-aware blending so NS term ramps in near the goal.

Requires: cvxpy (ECOS by default; OSQP optional).
"""

import numpy as np
import mujoco
from mujoco import viewer
import cvxpy as cp

# =========================
# User knobs
# =========================
XML_PATH      = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
ARM_JOINTS    = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
EE_BODY_NAME  = "hand"  # end-effector body name in MJCF

DT            = 1.0/300.0             # control dt (s)
K_POS         = 2.0                   # m/s per m position error
K_ANG         = 2.0                   # rad/s per rad orientation error
VEL_LIN_MAX   = 0.25                  # m/s clamp for task-space linear velocity
VEL_ANG_MAX   = 2.0                   # rad/s clamp for task-space angular velocity

# QP weights (stacked LS):  || S v - b ||^2  with S = [W_task*J;  W_ns*N;  sqrt(QP_REG)*I]
W_POS         = 1.0                   # weight for position tracking rows (first 3)
W_ANG         = 1.0                   # weight for orientation tracking rows (next 3)
QP_REG        = 1e-4                  # small Tikhonov regularization on qdot (via extra rows)

# Null-space manipulability (QP)
W_NS          = 1.0                   # null-space block weight (applied as sqrt(W_NS) inside S)
NS_GAIN       = 0.005                 # scale of manipulability-ascent velocity (try 0.002–0.02)
NS_CLAMP      = 1.0                   # clamp ||qdot_ns_des|| (rad/s); None to disable
NS_EVERY      = 5                     # compute FD gradient every K ticks
NS_DLS_LAMBDA = 1e-2                  # damping for DLS projector N = I - J^T (JJ^T+λI)^-1 J

# Convergence-aware blending: ns_alpha in [0,1] multiplies null-space block
NS_BLEND             = True
NS_BLEND_ERR_MULT    = 3.0             # ramp up null-space when err < this * TOL

POS_TOL       = 1e-3                  # m
ANG_TOL       = 1e-2                  # rad (angle-axis norm)

GRIPPER_OPEN  = 255.0                 # keep tendon/gripper open if present

# Trajectory knobs
TRAJ_TIME       = 10.0                # seconds start -> goal
USE_MINJERK     = True

# Solver choice
PREFER_OSQP   = False                 # True to try OSQP first (will auto-fallback to ECOS on trouble)

# ---- Target Pose (WORLD frame) ----
TARGET_POS_W    = np.array([0.25, 0.40, 0.45])
TARGET_RPY_DEG  = (0.0, 0.0, 0.0)
TARGET_QUAT_W   = None  # (w,x,y,z) overrides RPY if set


# =========================
# Math utilities
# =========================
def clamp_to_limits(q, model, joint_ids):
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        out[i] = np.clip(out[i], min(lo, hi), max(lo, hi))
    return out

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return np.array([1.0,0,0,0]) if n == 0 else q / n

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]])

def euler_zyx_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll*0.5),  np.sin(roll*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cy, sy = np.cos(yaw*0.5),   np.sin(yaw*0.5)
    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = cy*sp*cr + sy*cp*sr
    z = sy*cp*cr - cy*sp*sr
    return quat_normalize(np.array([w,x,y,z]))

def mat_to_quat(R):
    R = np.asarray(R).reshape(3,3)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2.0; w=0.25*s
        x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2.0; w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i == 1:
            s=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2.0; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2.0; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return quat_normalize(np.array([w,x,y,z]))

def quat_log_error(q_cur, q_des):
    q_cur = quat_normalize(q_cur); q_des = quat_normalize(q_des)
    q_err = quat_mul(q_des, quat_conj(q_cur))
    if q_err[0] < 0: q_err = -q_err
    w,x,y,z = q_err; v = np.array([x,y,z]); s = np.linalg.norm(v)
    if s < 1e-8: return 2.0 * v
    angle = 2.0 * np.arctan2(s, w); axis = v / s
    return angle * axis

def slerp(q0, q1, u):
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0: q1 = -q1; dot = -dot
    if dot > 0.9995: return quat_normalize(q0 + u*(q1-q0))
    th0 = np.arccos(dot); s0 = np.sin(th0)
    a = np.sin((1.0-u)*th0)/s0; b = np.sin(u*th0)/s0
    return a*q0 + b*q1

def minjerk(u): return u**3 * (10 - 15*u + 6*u*u)

def get_body_pose_world(data, bid):
    p = data.xpos[bid].copy() if hasattr(data, "xpos") else data.xipos[bid].copy()
    if hasattr(data, "xquat"): q = data.xquat[bid].copy()
    else:
        R = (data.xmat[bid] if hasattr(data,"xmat") else data.ximat[bid]).reshape(3,3)
        q = mat_to_quat(R)
    return p, q


# =========================
# Manipulability helpers
# =========================
def compute_body_J(model, data, ee_bid, daddr):
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
    return np.vstack([jacp[:, daddr], jacr[:, daddr]])  # 6x7

def dls_projector(J, lam):
    JJt = J @ J.T
    Jdag = J.T @ np.linalg.inv(JJt + (lam**2) * np.eye(J.shape[0]))
    return np.eye(J.shape[1]) - Jdag @ J

def logdet_manip(J, reg=1e-8):
    A = J @ J.T + reg * np.eye(J.shape[0])
    sign, logdet = np.linalg.slogdet(A)
    return logdet if sign > 0 else -1e12

def grad_logdet_manip_fd(model, data_fd, qaddr, daddr, ee_bid, q_base, eps=1e-4, reg=1e-8):
    n = len(qaddr); g = np.zeros(n)
    for i in range(n):
        # +eps
        data_fd.qpos[:] = 0.0
        data_fd.qpos[qaddr] = q_base; data_fd.qpos[qaddr[i]] = q_base[i] + eps
        mujoco.mj_forward(model, data_fd)
        f_plus = logdet_manip(compute_body_J(model, data_fd, ee_bid, daddr))
        # -eps
        data_fd.qpos[qaddr] = q_base; data_fd.qpos[qaddr[i]] = q_base[i] - eps
        mujoco.mj_forward(model, data_fd)
        f_minus = logdet_manip(compute_body_J(model, data_fd, ee_bid, daddr))
        g[i] = (f_plus - f_minus) / (2.0 * eps)
    data_fd.qpos[qaddr] = q_base; mujoco.mj_forward(model, data_fd)
    return g


# =========================
# Model setup
# =========================
def load_model_and_indices():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1: mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:            mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
    qaddr     = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    daddr     = np.array([model.jnt_dofadr[jid]  for jid in joint_ids], dtype=int)
    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    if ee_bid < 0: raise RuntimeError(f'Body "{EE_BODY_NAME}" not found in model.')
    return model, data, joint_ids, qaddr, daddr, ee_bid


# =========================
# QP builder (stacked LS)
# =========================
def build_qp_stacked(n, dt):
    """
    Minimize || S v - b ||^2  subject to  q + dt*v ∈ [q_lo, q_hi]
    Shapes:
      S: (6 + n + n, n)  ; rows = task (6) + null-space (n) + reg (n)
      b: (6 + n + n,)
    """
    v     = cp.Variable(n)
    S     = cp.Parameter((6 + n + n, n))
    b     = cp.Parameter(6 + n + n)
    q_now = cp.Parameter(n)
    qlo   = cp.Parameter(n)
    qhi   = cp.Parameter(n)

    obj = cp.sum_squares(S @ v - b)
    cons = [ qlo <= q_now + dt * v, q_now + dt * v <= qhi ]
    prob = cp.Problem(cp.Minimize(obj), cons)

    params = dict(v=v, S=S, b=b, q_now=q_now, qlo=qlo, qhi=qhi)
    return prob, params


# =========================
# Main loop
# =========================
def run():
    model, data, joint_ids, qaddr, daddr, ee_bid = load_model_and_indices()
    n = len(joint_ids)

    # Target quaternion
    if TARGET_QUAT_W is None:
        roll, pitch, yaw = np.deg2rad(TARGET_RPY_DEG)
        q_goal = euler_zyx_to_quat(roll, pitch, yaw)
    else:
        q_goal = quat_normalize(TARGET_QUAT_W)
    p_goal = TARGET_POS_W.astype(float).copy()

    # Initial state
    q_ref = data.qpos[qaddr].copy()
    p_start, q_start = get_body_pose_world(data, ee_bid)

    # Joint limits
    q_lo = np.array([model.jnt_range[jid][0] for jid in joint_ids])
    q_hi = np.array([model.jnt_range[jid][1] for jid in joint_ids])

    # QP setup
    prob, prm = build_qp_stacked(n, DT)

    # Weighting matrices (constant)
    W_task = np.diag([np.sqrt(W_POS)]*3 + [np.sqrt(W_ANG)]*3)
    S_reg  = np.sqrt(QP_REG) * np.eye(n)
    b_reg  = np.zeros(n)

    # Scratch data for FD gradient
    data_fd = mujoco.MjData(model)

    # Gripper
    if model.nu >= 8: data.ctrl[7] = GRIPPER_OPEN

    # Timing
    t0 = data.time
    printed_converged = False
    printed_traj_done = False
    tick = 0
    last_grad = np.zeros(n)

    print("\n=== Pose IK (QP trajectory -> hold) + Null-space Manipulability (stacked LS) ===")
    print(f"TRAJ_TIME = {TRAJ_TIME}s, easing = {'minjerk' if USE_MINJERK else 'linear'}\n")

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            # Progress
            u = (data.time - t0) / TRAJ_TIME
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
            s = minjerk(u) if USE_MINJERK else u

            # Desired pose
            p_des = (1.0 - s) * p_start + s * p_goal
            q_des = slerp(q_start, q_goal, s)

            if (not printed_traj_done) and (u >= 1.0):
                printed_traj_done = True
                print("[Trajectory complete] Holding pose and optimizing manipulability in null-space.")

            # Current pose and errors
            p_cur, q_cur = get_body_pose_world(data, ee_bid)
            e_pos = (p_des - p_cur)
            e_ang = quat_log_error(q_cur, q_des)

            pos_err_norm = float(np.linalg.norm(e_pos))
            ang_err_norm = float(np.linalg.norm(e_ang))
            if (not printed_converged) and (pos_err_norm <= POS_TOL) and (ang_err_norm <= ANG_TOL):
                printed_converged = True
                print(f"[Converged] |e_pos|={pos_err_norm:.3e}, |e_ang|={ang_err_norm:.3e}")

            # Desired twist (clamped)
            v_des = K_POS * e_pos
            w_des = K_ANG * e_ang
            vn = np.linalg.norm(v_des)
            if vn > VEL_LIN_MAX and vn > 0: v_des *= (VEL_LIN_MAX / vn)
            wn = np.linalg.norm(w_des)
            if wn > VEL_ANG_MAX and wn > 0: w_des *= (VEL_ANG_MAX / wn)
            xd = np.hstack([v_des, w_des])

            # Jacobian and DLS null-space projector
            J = compute_body_J(model, data, ee_bid, daddr)
            N = dls_projector(J, NS_DLS_LAMBDA)

            # Manipulability gradient (periodic FD)
            if tick % NS_EVERY == 0:
                last_grad = grad_logdet_manip_fd(
                    model, data_fd, qaddr, daddr, ee_bid,
                    q_base=q_ref, eps=1e-4, reg=1e-8
                )

            # Desired null-space velocity
            qdot_ns_des = NS_GAIN * last_grad
            if NS_CLAMP is not None:
                nrm = np.linalg.norm(qdot_ns_des)
                if nrm > NS_CLAMP and nrm > 0:
                    qdot_ns_des *= NS_CLAMP / nrm
            vns = N @ qdot_ns_des

            # Convergence-aware blending alpha
            if NS_BLEND:
                err_ratio = max(pos_err_norm/(NS_BLEND_ERR_MULT*POS_TOL),
                                ang_err_norm/(NS_BLEND_ERR_MULT*ANG_TOL))
                ns_alpha = float(np.clip(1.0 - err_ratio, 0.0, 1.0))
            else:
                ns_alpha = 1.0

            # --- Build stacked LS (constant shapes) ---
            # Task block
            S_task = W_task @ J
            b_task = W_task @ xd
            # Null-space block (fold weights numerically; keep Parameter structure simple)
            ns_scale = np.sqrt(W_NS) * np.sqrt(ns_alpha)
            S_ns = ns_scale * N
            b_ns = ns_scale * vns
            # Stack S and b
            S_stack = np.vstack([S_task, S_ns, S_reg])
            b_stack = np.hstack([b_task, b_ns, b_reg])

            # Update parameters
            prm["S"].value     = S_stack
            prm["b"].value     = b_stack
            prm["q_now"].value = q_ref
            prm["qlo"].value   = q_lo
            prm["qhi"].value   = q_hi

            # Solve QP
            solved = False
            if PREFER_OSQP:
                try:
                    prob.solve(solver=cp.OSQP, warm_start=False, verbose=False)
                    solved = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                except Exception:
                    solved = False
            if not solved:
                prob.solve(solver=cp.ECOS, warm_start=False, verbose=False)

            if prm["v"].value is None:
                qdot = np.zeros(n)
            else:
                qdot = np.array(prm["v"].value).reshape(-1)

            # Integrate & clamp; command
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8: data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            v.sync()
            tick += 1

    # Final report
    mujoco.mj_forward(model, data)
    p_final, q_final = get_body_pose_world(data, ee_bid)
    e_pos_f = p_goal - p_final
    e_ang_f = quat_log_error(q_final, q_goal)
    print("\nFinal pose error (vs goal):")
    print(f"  |pos| = {np.linalg.norm(e_pos_f):.6f} m,  e_pos = {e_pos_f}")
    print(f"  |ang| = {np.linalg.norm(e_ang_f):.6f} rad, e_ang = {e_ang_f}")
    print("Viewer closed. Exiting.")


if __name__ == "__main__":
    run()
