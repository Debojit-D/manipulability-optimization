#!/usr/bin/env python3
"""
DLS Pose IK + (optional) null-space manipulability ascent with post-goal hold optimization.
Runs twice (A/B): null-space DISABLED vs ENABLED, logs manipulability,
and plots both curves on the same figure.

- Tracking to a world-frame pose (linear pos interp + SLERP orient).
- Null-space task: gradient-ascent on log det(J J^T).
- Optional "hold optimization": after reaching the goal, keep the EE fixed and
  continue pure null-space ascent to improve final posture.

Outputs:
- logs/ik_ns_logs_disabled.csv  (NS off)
- logs/ik_ns_logs_enabled.csv   (NS on)
- logs/manipulability_compare.png (overlayed plot)
"""

import os, pickle
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# =========================
# User knobs
# =========================
XML_PATH      = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
ARM_JOINTS    = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
EE_BODY_NAME  = "hand"  # end-effector body name in MJCF

DT            = 1.0/300.0             # control dt (s)
LAMBDA        = 1e-2                  # DLS damping
K_POS         = 2.0                   # m/s per m position error
K_ANG         = 2.0                   # rad/s per rad orientation error
VEL_LIN_MAX   = 0.25                  # m/s clamp
VEL_ANG_MAX   = 2.0                   # rad/s clamp

POS_TOL       = 1e-3                  # m
ANG_TOL       = 1e-2                  # rad (angle-axis norm)

GRIPPER_OPEN  = 255.0                 # keep tendon/gripper open if present

# ---- Trajectory knobs ----
TRAJ_TIME       = 10.0                # seconds to go from start -> goal
USE_MINJERK     = True                # smooth easing (5th-order); else linear
HOLD_TIME       = 10.0                 # seconds to hold after reaching goal (for plots/logs)

# ---- Post-goal null-space optimization ("hold optimize") ----
HOLD_OPTIMIZE        = True           # keep optimizing after reaching the goal
HOLD_TASK_SCALE      = 0.2            # reduce task twist during hold so NS can work
K_HOLD_POS           = 6.0            # stiffer gains at the goal (to keep pose pinned)
K_HOLD_ANG           = 6.0

# ---- Null-space manipulability maximization ----
NS_GAIN              = 0.5            # base scale of manipulability-ascent velocity
NS_EPS_FD            = 1e-4           # finite-difference step [rad] for grad of logdet
MANIP_LOGDET_REG     = 1e-8           # small I regularization in JJ^T before logdet
NS_CLAMP             = 5.0            # clamp ||qdot_ns_des|| (rad/s) before projection; None disables
NS_EVERY             = 5              # compute gradient every K ticks (reuse in between) for speed

# Gain scheduling: grow NS influence as we approach the goal (s in [0,1])
def ns_gain_schedule(s: float) -> float:
    return 0.2 + 0.8*(s**2)           # small early, strong near goal

# Null-space backtracking line-search (to ensure uphill steps)
NS_LS_ENABLED       = True
NS_LS_MAX_SHRINK    = 5               # number of backtracking steps
NS_LS_FACTOR        = 0.5             # shrink factor per backtrack

# ---- Target Pose (WORLD frame) ----
# TARGET_POS_W    = np.array([0.25, 0.40, 0.45])      # meters (x,y,z in world)
# TARGET_RPY_DEG  = (0.0, 20.0, 0.0)                  # roll, pitch, yaw in degrees
# ---- Target Pose (WORLD frame) ----
# ---- Target Pose (WORLD frame) ----
# ---- Target Pose (WORLD frame) ----
TARGET_POS_W    = np.array([0.20, -0.20, 0.4])      # meters (x,y,z in world)
TARGET_RPY_DEG  = (-45.0, 35.0, -20.0)                   # roll, pitch, yaw in degrees
TARGET_QUAT_W   = None  # if set (w,x,y,z), overrides RPY. e.g., np.array([0.7071, 0,0,0.7071])

# ---- Logging / plotting ----
HEADLESS        = True
SAVE_LOGS       = True
LOG_DIR         = "logs"


# =========================
# Math utilities
# =========================
def clamp_to_limits(q, model, joint_ids):
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        if np.isfinite(lo) and np.isfinite(hi):
            out[i] = np.clip(out[i], min(lo, hi), max(lo, hi))
    return out

def dls_pseudoinverse(J, lam):
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + (lam**2) * np.eye(J.shape[0]))

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

def euler_zyx_to_quat(roll, pitch, yaw):
    cr = np.cos(roll*0.5);  sr = np.sin(roll*0.5)
    cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
    cy = np.cos(yaw*0.5);   sy = np.sin(yaw*0.5)
    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = cy*sp*cr + sy*cp*sr
    z = sy*cp*cr - cy*sp*sr
    return quat_normalize(np.array([w,x,y,z]))

def mat_to_quat(R):
    R = np.asarray(R).reshape(3,3)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2.0
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])*2.0
            w = (R[2,1]-R[1,2])/s
            x = 0.25*s
            y = (R[0,1]+R[1,0])/s
            z = (R[0,2]+R[2,0])/s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])*2.0
            w = (R[0,2]-R[2,0])/s
            x = (R[0,1]+R[1,0])/s
            y = 0.25*s
            z = (R[1,2]+R[2,1])/s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])*2.0
            w = (R[1,0]-R[0,1])/s
            x = (R[0,2]+R[2,0])/s
            y = (R[1,2]+R[2,1])/s
            z = 0.25*s
    return quat_normalize(np.array([w,x,y,z]))

def quat_log_error(q_cur, q_des):
    q_cur = quat_normalize(q_cur)
    q_des = quat_normalize(q_des)
    q_err = quat_mul(q_des, quat_conj(q_cur))
    if q_err[0] < 0.0:
        q_err = -q_err
    w, x, y, z = q_err
    v = np.array([x, y, z])
    s = np.linalg.norm(v)
    if s < 1e-8:
        return 2.0 * v
    angle = 2.0 * np.arctan2(s, w)
    axis  = v / s
    return angle * axis

def slerp(q0, q1, u):
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + u*(q1 - q0)
        return quat_normalize(q)
    theta0 = np.arccos(dot)
    sin0   = np.sin(theta0)
    s0 = np.sin((1.0 - u) * theta0) / sin0
    s1 = np.sin(u * theta0) / sin0
    return s0 * q0 + s1 * q1

def minjerk(u): return u**3 * (10 - 15*u + 6*u*u)

def get_body_pose_world(data, bid):
    if hasattr(data, "xpos"):
        p = data.xpos[bid].copy()
    elif hasattr(data, "xipos"):
        p = data.xipos[bid].copy()
    else:
        raise RuntimeError("MjData has no xpos/xipos for bodies.")
    if hasattr(data, "xquat"):
        q = data.xquat[bid].copy()
    else:
        if hasattr(data, "xmat"):
            R = data.xmat[bid].reshape(3,3)
        elif hasattr(data, "ximat"):
            R = data.ximat[bid].reshape(3,3)
        else:
            raise RuntimeError("MjData has no xquat/xmat/ximat for bodies.")
        q = mat_to_quat(R)
    return p, q

# =========================
# Manipulability helpers
# =========================
def compute_body_J(model, data, ee_bid, daddr):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
    return np.vstack([jacp[:, daddr], jacr[:, daddr]])  # 6x7

def logdet_manip(J, reg=1e-8):
    A = J @ J.T + reg * np.eye(J.shape[0])
    sign, logdet = np.linalg.slogdet(A)
    return logdet if sign > 0 else -1e12

def grad_logdet_manip_fd(model, data_fd, qaddr, daddr, ee_bid, q_base, eps=1e-4, reg=1e-8):
    n = len(qaddr)
    g = np.zeros(n)
    # Clean baseline
    data_fd.qpos[:] = 0.0
    data_fd.qpos[qaddr] = q_base
    mujoco.mj_forward(model, data_fd)
    for i in range(n):
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

def nullspace_projector(J, J_pinv):
    n = J_pinv.shape[0]
    return np.eye(n) - J_pinv @ J

def ns_step_with_linesearch(model, data_fd, qaddr, daddr, ee_bid,
                            q_ref, N, qdot_ns_des, dt, reg):
    """Backtrack in null-space direction so logdet(JJ^T) increases."""
    # Baseline value at q_ref
    data_fd.qpos[:] = 0.0
    data_fd.qpos[qaddr] = q_ref
    mujoco.mj_forward(model, data_fd)
    f0 = logdet_manip(compute_body_J(model, data_fd, ee_bid, daddr), reg=reg)

    scale = 1.0
    best_scale = 0.0
    best_val = -np.inf
    for _ in range(NS_LS_MAX_SHRINK + 1):
        q_try = q_ref + dt * (N @ (scale * qdot_ns_des))
        data_fd.qpos[qaddr] = q_try
        mujoco.mj_forward(model, data_fd)
        f1 = logdet_manip(compute_body_J(model, data_fd, ee_bid, daddr), reg=reg)
        if f1 > f0:
            best_scale = scale
            best_val = f1
            break
        if f1 > best_val:
            best_val = f1; best_scale = scale
        scale *= NS_LS_FACTOR
    return best_scale

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
    if ee_bid < 0:
        raise RuntimeError(f'Body "{EE_BODY_NAME}" not found in model.')

    return model, data, joint_ids, qaddr, daddr, ee_bid

# =========================
# One run (NS enabled/disabled)
# =========================
def run_once(enable_nullspace: bool, label: str):
    model, data, joint_ids, qaddr, daddr, ee_bid = load_model_and_indices()
    data_fd = mujoco.MjData(model)

    # Target
    if TARGET_QUAT_W is None:
        roll, pitch, yaw = np.deg2rad(TARGET_RPY_DEG)
        q_goal = euler_zyx_to_quat(roll, pitch, yaw)
    else:
        q_goal = quat_normalize(TARGET_QUAT_W)
    p_goal = TARGET_POS_W.astype(float).copy()

    # Init
    q_ref = data.qpos[qaddr].copy()
    p_start, q_start = get_body_pose_world(data, ee_bid)
    if model.nu >= 8:
        data.ctrl[7] = GRIPPER_OPEN

    logs = {"t": [], "pos_err": [], "ang_err": [], "manip_logdet": [], "yoshikawa": [], "grad_norm": []}

    t0 = data.time
    printed_traj_done = False
    tick = 0
    last_grad = np.zeros_like(q_ref)

    def step_body():
        nonlocal q_ref, last_grad, printed_traj_done, tick
        mujoco.mj_forward(model, data)

        # progress
        u = (data.time - t0) / TRAJ_TIME
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        s = minjerk(u) if USE_MINJERK else u

        # phases
        in_traj = (u < 1.0)
        time_since_goal = max(0.0, data.time - (t0 + TRAJ_TIME))
        in_hold  = (u >= 1.0) and (time_since_goal <= HOLD_TIME)
        in_opt   = in_hold and HOLD_OPTIMIZE

        # desired pose (freeze at goal after u>=1)
        p_des = (1.0 - s) * p_start + s * p_goal
        q_des = slerp(q_start, q_goal, s)

        if (not printed_traj_done) and (u >= 1.0):
            printed_traj_done = True
            print(f"[{label}] Trajectory complete -> holding (opt={HOLD_OPTIMIZE}).")

        # current pose
        p_cur, q_cur = get_body_pose_world(data, ee_bid)

        # errors
        e_pos = (p_des - p_cur)
        e_ang = quat_log_error(q_cur, q_des)
        pos_err_norm = np.linalg.norm(e_pos)
        ang_err_norm = np.linalg.norm(e_ang)

        # gains: normal vs hold (tighter at goal)
        kpos = K_POS if in_traj else K_HOLD_POS
        kang = K_ANG if in_traj else K_HOLD_ANG

        # twist (clamped). During hold-opt, reduce task authority.
        v_des = kpos * e_pos
        w_des = kang * e_ang
        if in_opt:
            v_des *= HOLD_TASK_SCALE
            w_des *= HOLD_TASK_SCALE

        v_norm = np.linalg.norm(v_des)
        if v_norm > VEL_LIN_MAX and v_norm > 0.0:
            v_des *= (VEL_LIN_MAX / v_norm)
        w_norm = np.linalg.norm(w_des)
        if w_norm > VEL_ANG_MAX and w_norm > 0.0:
            w_des *= (VEL_ANG_MAX / w_norm)
        xd = np.hstack([v_des, w_des])

        # Jacobian & metrics
        J = compute_body_J(model, data, ee_bid, daddr)
        JJt = J @ J.T + MANIP_LOGDET_REG * np.eye(6)
        sign, logdet = np.linalg.slogdet(JJt)
        manip_logdet = logdet if sign > 0 else -1e12
        yoshikawa = float(np.exp(0.5 * manip_logdet)) if sign > 0 else 0.0

        # IK
        J_pinv = dls_pseudoinverse(J, LAMBDA)
        qdot_task = J_pinv @ xd

        # NS ascent (scheduled + optional backtracking)
        eff_ns_gain = 0.0
        if enable_nullspace:
            eff_ns_gain = NS_GAIN * ns_gain_schedule(s)
            if in_opt:
                # ensure decent authority in hold optimize phase
                eff_ns_gain = max(eff_ns_gain, NS_GAIN)

        if tick % NS_EVERY == 0:
            last_grad = grad_logdet_manip_fd(
                model, data_fd, qaddr, daddr, ee_bid,
                q_base=q_ref, eps=NS_EPS_FD, reg=MANIP_LOGDET_REG
            )
        qdot_ns_des = eff_ns_gain * last_grad
        if NS_CLAMP is not None:
            nrm = np.linalg.norm(qdot_ns_des)
            if nrm > NS_CLAMP and nrm > 0:
                qdot_ns_des *= NS_CLAMP / nrm

        N = nullspace_projector(J, J_pinv)

        if enable_nullspace and NS_LS_ENABLED:
            scale = ns_step_with_linesearch(model, data_fd, qaddr, daddr, ee_bid,
                                            q_ref, N, qdot_ns_des, DT, MANIP_LOGDET_REG)
            qdot_ns_des *= scale

        # Compose command
        qdot = qdot_task + (N @ qdot_ns_des)

        # integrate + command
        q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
        data.ctrl[0:7] = q_ref
        if model.nu >= 8:
            data.ctrl[7] = GRIPPER_OPEN

        # log
        logs["t"].append(data.time)
        logs["pos_err"].append(pos_err_norm)
        logs["ang_err"].append(ang_err_norm)
        logs["manip_logdet"].append(manip_logdet)
        logs["yoshikawa"].append(yoshikawa)
        logs["grad_norm"].append(float(np.linalg.norm(last_grad)))

        mujoco.mj_step(model, data)
        tick += 1

    print(f"\n=== RUN: {label} — Null-space {'ENABLED' if enable_nullspace else 'DISABLED'} (NS_GAIN base={NS_GAIN}) ===")
    if HEADLESS:
        # Run a fixed horizon: TRAJ_TIME + HOLD_TIME
        T_end = TRAJ_TIME + HOLD_TIME
        while (data.time - t0) < T_end:
            step_body()
    else:
        with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
            while v.is_running():
                step_body()
                v.sync()

    # Final report
    mujoco.mj_forward(model, data)
    p_final, q_final = get_body_pose_world(data, ee_bid)
    e_pos_f = TARGET_POS_W - p_final
    target_q = (euler_zyx_to_quat(*np.deg2rad(TARGET_RPY_DEG)) if TARGET_QUAT_W is None else quat_normalize(TARGET_QUAT_W))
    e_ang_f = quat_log_error(q_final, target_q)
    print(f"[{label}] Final |pos_err|={np.linalg.norm(e_pos_f):.6f} m, |ang_err|={np.linalg.norm(e_ang_f):.6f} rad")

    return logs

# =========================
# Plot compare (Yoshikawa w only, larger & thicker)
# =========================
def plot_compare(L_off, L_on, figsize=(11, 6), dpi=220, lw=2.8, fs=14):
    t_off = np.array(L_off["t"], dtype=float)
    t_on  = np.array(L_on["t"],  dtype=float)

    # Bump font sizes globally for this figure
    plt.rcParams.update({
        "font.size": fs,
        "axes.labelsize": fs,
        "axes.titlesize": fs + 2,
        "legend.fontsize": fs - 1,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
    })

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Yoshikawa w curves (thicker lines)
    ax.plot(t_off, L_off["yoshikawa"], label="Yoshikawa w — NS OFF", linewidth=lw)
    ax.plot(t_on,  L_on["yoshikawa"],  label="Yoshikawa w — NS ON",  linewidth=lw)

    # Mark when the goal is reached
    ax.axvline(TRAJ_TIME, linestyle=":", color="k", linewidth=1.2, alpha=0.6)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("Yoshikawa w")
    ax.set_title("Manipulability (Yoshikawa w): NS OFF vs NS ON")
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.legend(loc="best")

    fig.tight_layout()
    if SAVE_LOGS:
        os.makedirs(LOG_DIR, exist_ok=True)
        fig_path = os.path.join(LOG_DIR, "manipulability_w_compare.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {os.path.abspath(fig_path)}")
    plt.show()


def save_logs_csv(logs, path_csv):
    with open(path_csv, "w") as f:
        f.write("t,pos_err,ang_err,manip_logdet,yoshikawa,grad_norm\n")
        for i in range(len(logs["t"])):
            f.write(f"{logs['t'][i]},{logs['pos_err'][i]},{logs['ang_err'][i]},"
                    f"{logs['manip_logdet'][i]},{logs['yoshikawa'][i]},{logs['grad_norm'][i]}\n")

# =========================
# Entry
# =========================
def main():
    if SAVE_LOGS:
        os.makedirs(LOG_DIR, exist_ok=True)

    # A: null-space disabled
    L_off = run_once(enable_nullspace=False, label="NS_OFF")

    # B: null-space enabled
    L_on  = run_once(enable_nullspace=True,  label="NS_ON")

    # Save per-run logs
    if SAVE_LOGS:
        with open(os.path.join(LOG_DIR, "ik_ns_logs_disabled.pkl"), "wb") as f:
            pickle.dump(L_off, f)
        with open(os.path.join(LOG_DIR, "ik_ns_logs_enabled.pkl"), "wb") as f:
            pickle.dump(L_on, f)
        save_logs_csv(L_off, os.path.join(LOG_DIR, "ik_ns_logs_disabled.csv"))
        save_logs_csv(L_on,  os.path.join(LOG_DIR, "ik_ns_logs_enabled.csv"))
        print(f"Logs saved to: {os.path.abspath(LOG_DIR)}")

    # Combined figure
    plot_compare(L_off, L_on)

if __name__ == "__main__":
    main()
