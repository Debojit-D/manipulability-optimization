#!/usr/bin/env python3
"""
Pose IK with QP (world-frame target) — Run twice:
1) without manipulability optimization
2) with manipulability optimization (smoothly blended)

After closing the first viewer, the second run starts.
After closing the second viewer, comparison plots are shown.

Requires: numpy, mujoco, cvxpy, matplotlib
Solver: tries OSQP, then ECOS, then SCS.
"""

import numpy as np
import mujoco
from mujoco import viewer
import cvxpy as cp
import matplotlib.pyplot as plt

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

# QP weights (stacked LS):  || S v - b ||^2  with S = [W_task*J;  sqrt(W_NS*ns_alpha)*N;  sqrt(QP_REG)*I]
W_POS         = 1.0                   # weight for position tracking rows (first 3)
W_ANG         = 1.0                   # weight for orientation tracking rows (next 3)
QP_REG        = 1e-4                  # small Tikhonov regularization on qdot (via extra rows)

# Null-space manipulability (QP)
W_NS          = 0.5                   # null-space block weight (soft vs task)
NS_GAIN       = 0.02                 # scale of manipulability ascent velocity (0.002–0.01 typical)
NS_CLAMP      = 0.10                   # clamp ||qdot_ns_des|| (rad/s); None to disable
NS_EVERY      = 2                     # compute FD gradient every K ticks
NS_DLS_LAMBDA = 3e-2                  # damping for DLS projector N = I - J^T (JJ^T+λI)^-1 J

# Convergence-aware blending & smoothing
NS_BLEND_ERR_MULT = 150.0               # start enabling NS when errors < 3× tolerances (sooner if larger)
NS_ALPHA_RATE     = 5.0               # per-second max change of ns_alpha (rate limit)
VNS_SMOOTH        = 0.5               # EMA smoothing for null-space command (0..1, higher=faster)

POS_TOL       = 1e-3                  # m
ANG_TOL       = 1e-2                  # rad (angle-axis norm)

GRIPPER_OPEN  = 255.0                 # keep tendon/gripper open if present

# Trajectory knobs
TRAJ_TIME       = 10.0                # seconds start -> goal
USE_MINJERK     = True

# Target Pose (WORLD frame)
TARGET_POS_W    = np.array([0.35, -0.40, 0.35])
TARGET_RPY_DEG  = (10.0, 20.0, 0.0)
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

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)

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
    return logdet if sign > 0 else -1e-12

def grad_logdet_manip_fd(model, data_fd, qaddr, daddr, ee_bid, q_base, eps=1e-4, reg=1e-8):
    n = len(qaddr); g = np.zeros(n)
    # start each FD from the *current* full state for stability
    for i in range(n):
        data_fd.qpos[:] = data_fd.qpos[:]  # no-op to keep shape
        data_fd.qpos[:] = 0.0
        data_fd.qpos[qaddr] = q_base
        data_fd.qpos[qaddr[i]] = q_base[i] + eps
        mujoco.mj_forward(model, data_fd)
        f_plus = logdet_manip(compute_body_J(model, data_fd, ee_bid, daddr), reg=reg)

        data_fd.qpos[qaddr] = q_base
        data_fd.qpos[qaddr[i]] = q_base[i] - eps
        mujoco.mj_forward(model, data_fd)
        f_minus = logdet_manip(compute_body_J(model, data_fd, ee_bid, daddr), reg=reg)

        g[i] = (f_plus - f_minus) / (2.0 * eps)

    data_fd.qpos[qaddr] = q_base
    mujoco.mj_forward(model, data_fd)
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

def solve_qp(prob):
    # Try OSQP -> ECOS -> SCS
    try:
        prob.solve(solver=cp.OSQP, warm_start=False, verbose=False)
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE): return
    except Exception:
        pass
    try:
        prob.solve(solver=cp.ECOS, warm_start=False, verbose=False)
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE): return
    except Exception:
        pass
    prob.solve(solver=cp.SCS, warm_start=False, verbose=False)

# =========================
# One full session (viewer loop)
# =========================
def run_session(enable_manip_opt: bool, label: str):
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

    # Logs
    log_t, log_pe, log_ae, log_m, log_nsa = [], [], [], [], []

    # Timing & smoothing
    t0 = data.time
    printed_traj_done = False
    tick = 0
    last_grad = np.zeros(n)
    ns_alpha_prev = 0.0
    vns_prev = np.zeros(n)

    print(f"\n=== {label} ===")
    print(f"Manipulability optimization: {'ON' if enable_manip_opt else 'OFF'}")
    print(f"TRAJ_TIME = {TRAJ_TIME}s, easing = {'minjerk' if USE_MINJERK else 'linear'}")

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            # Progress 0..1
            u = (data.time - t0) / TRAJ_TIME
            u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
            s = minjerk(u) if USE_MINJERK else u

            # Desired pose
            p_des = (1.0 - s) * p_start + s * p_goal
            q_des = slerp(q_start, q_goal, s)

            if (not printed_traj_done) and (u >= 1.0):
                printed_traj_done = True
                print("[Trajectory complete] Holding final pose.")

            # Current pose and errors
            p_cur, q_cur = get_body_pose_world(data, ee_bid)
            e_pos = (p_des - p_cur)
            e_ang = quat_log_error(q_cur, q_des)

            pos_err_norm = float(np.linalg.norm(e_pos))
            ang_err_norm = float(np.linalg.norm(e_ang))

            # Desired twist (clamped)
            v_des = K_POS * e_pos
            w_des = K_ANG * e_ang
            vn = np.linalg.norm(v_des)
            if vn > VEL_LIN_MAX and vn > 0: v_des *= (VEL_LIN_MAX / vn)
            wn = np.linalg.norm(w_des)
            if wn > VEL_ANG_MAX and wn > 0: w_des *= (VEL_ANG_MAX / wn)
            xd = np.hstack([v_des, w_des])

            # Jacobian and projector
            J = compute_body_J(model, data, ee_bid, daddr)
            N = dls_projector(J, NS_DLS_LAMBDA)

            # Manipulability gradient (periodic FD)
            if enable_manip_opt and tick % NS_EVERY == 0:
                last_grad = grad_logdet_manip_fd(
                    model, data_fd, qaddr, daddr, ee_bid,
                    q_base=q_ref, eps=1e-4, reg=1e-8
                )

            # Desired null-space velocity (normalized, clamped)
            if enable_manip_opt:
                g = last_grad
                g_norm = np.linalg.norm(g)
                if g_norm > 1e-8: g = g / g_norm
                qdot_ns_des = NS_GAIN * g
                if NS_CLAMP is not None:
                    qdot_ns_des *= min(1.0, NS_CLAMP/(np.linalg.norm(qdot_ns_des)+1e-8))
                vns_target = N @ qdot_ns_des
            else:
                vns_target = np.zeros(n)

            # Blending alpha (time gate + error gate), then rate-limit + smooth vns
            if enable_manip_opt:
                time_gate = smoothstep(s)**2
                err_ratio = max(pos_err_norm/(NS_BLEND_ERR_MULT*POS_TOL + 1e-12),
                                ang_err_norm/(NS_BLEND_ERR_MULT*ANG_TOL + 1e-12))
                error_gate = smoothstep(1.0 - np.clip(err_ratio, 0.0, 1.0))
                ns_alpha_target = min(time_gate, error_gate)
                # rate limit
                step = NS_ALPHA_RATE * DT
                ns_alpha = ns_alpha_prev + np.clip(ns_alpha_target - ns_alpha_prev, -step, step)
            else:
                ns_alpha = 0.0

            ns_alpha_prev = ns_alpha
            vns = (1.0 - VNS_SMOOTH) * vns_prev + VNS_SMOOTH * vns_target
            vns_prev = vns

            # --- Build stacked LS (constant shapes) ---
            # Task block
            S_task = W_task @ J
            b_task = W_task @ xd
            # Null-space block
            ns_scale = np.sqrt(W_NS) * np.sqrt(max(ns_alpha, 0.0))
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
            solve_qp(prob)
            if prm["v"].value is None:
                qdot = np.zeros(n)
            else:
                qdot = np.array(prm["v"].value).reshape(-1)

            # Integrate & clamp; command
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8: data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            # Logs
            log_t.append(float(data.time - t0))
            log_pe.append(pos_err_norm)
            log_ae.append(ang_err_norm)
            log_m.append(logdet_manip(J))
            log_nsa.append(float(ns_alpha))
            v.sync()
            tick += 1

    # Final report after closing viewer
    mujoco.mj_forward(model, data)
    p_final, q_final = get_body_pose_world(data, ee_bid)
    e_pos_f = p_goal - p_final
    e_ang_f = quat_log_error(q_final, q_goal)
    print("\nFinal pose error (vs goal):")
    print(f"  |pos| = {np.linalg.norm(e_pos_f):.6f} m,  e_pos = {e_pos_f}")
    print(f"  |ang| = {np.linalg.norm(e_ang_f):.6f} rad, e_ang = {e_ang_f}")

    logs = {
        "label": label,
        "t": np.array(log_t),
        "pos_err": np.array(log_pe),
        "ang_err": np.array(log_ae),
        "manip_logdet": np.array(log_m),
        "ns_alpha": np.array(log_nsa),
    }
    return logs

# =========================
# Main: run twice then plot
# =========================
def main():
    # 1) Baseline (no manipulability)
    logs_no = run_session(enable_manip_opt=False, label="Run A: Tracking Only")

    # 2) Optimized (manipulability on)
    logs_yes = run_session(enable_manip_opt=True, label="Run B: Tracking + Null-space Manipulability")

    # 3) Comparison plots
    fig, axs = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)

    axs[0].plot(logs_no["t"], logs_no["pos_err"], label="No NS")
    axs[0].plot(logs_yes["t"], logs_yes["pos_err"], label="With NS")
    axs[0].set_title("Position error norm (m)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("||e_pos||")
    axs[0].grid(True); axs[0].legend()

    axs[1].plot(logs_no["t"], logs_no["ang_err"], label="No NS")
    axs[1].plot(logs_yes["t"], logs_yes["ang_err"], label="With NS")
    axs[1].set_title("Orientation error norm (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("||e_ang||")
    axs[1].grid(True); axs[1].legend()

    axs[2].plot(logs_no["t"], logs_no["manip_logdet"], label="No NS")
    axs[2].plot(logs_yes["t"], logs_yes["manip_logdet"], label="With NS")
    axs[2].set_title("Manipulability: log det(J Jᵀ)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("logdet")
    axs[2].grid(True); axs[2].legend()

    plt.show()

if __name__ == "__main__":
    main()
