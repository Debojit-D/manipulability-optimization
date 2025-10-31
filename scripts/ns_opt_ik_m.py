#!/usr/bin/env python3
"""
Pose IK with DLS (world-frame target) — trajectory to target + hold + null-space manipulability maximization.

- Interpolates from current EE pose to target (world) over TRAJ_TIME seconds.
- Position uses linear interpolation; orientation uses SLERP.
- After reaching the target, continues to hold it indefinitely.
- Null-space task maximizes Yoshikawa manipulability via gradient ascent on log det(J J^T).

xd = [vx, vy, vz, wx, wy, wz]; orientation error uses quaternion log map.
"""

import numpy as np
import mujoco
from mujoco import viewer

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

# ---- Null-space manipulability maximization ----
NS_GAIN              = 0.5         # <— scale of manipulability-ascent velocity (0.2–1.5 typical)
NS_EPS_FD            = 1e-4           # finite-difference step [rad] for gradient of logdet
MANIP_LOGDET_REG     = 1e-8           # small I regularization in JJ^T before logdet
NS_CLAMP             = 1.0            # clamp ||qdot_ns_des|| (rad/s) before projection; set None to disable
NS_EVERY             = 5              # compute gradient every K ticks (reuse in between) for speed

# ---- Target Pose (WORLD frame) ----
TARGET_POS_W    = np.array([0.25, 0.40, 0.45])      # meters (x,y,z in world)
TARGET_RPY_DEG  = (0.0, 20.0, 0.0)                   # roll, pitch, yaw in degrees
TARGET_QUAT_W   = None  # if set (w,x,y,z), overrides RPY. e.g., np.array([0.7071, 0,0,0.7071])


# =========================
# Math utilities
# =========================
def clamp_to_limits(q, model, joint_ids):
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
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
    # Rz(yaw)*Ry(pitch)*Rx(roll)
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
    # q_err = q_des * conj(q_cur), hemisphere (w>=0), return angle*axis (3,)
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
    """Spherical linear interpolation between quats q0->q1 at fraction u∈[0,1]."""
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    dot = np.dot(q0, q1)
    # Hemisphere fix
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    # If very close, fall back to lerp
    if dot > 0.9995:
        q = q0 + u*(q1 - q0)
        return quat_normalize(q)
    theta0 = np.arccos(dot)
    sin0   = np.sin(theta0)
    s0 = np.sin((1.0 - u) * theta0) / sin0
    s1 = np.sin(u * theta0) / sin0
    return s0 * q0 + s1 * q1

def minjerk(u):
    """5th-order time-scaling (0->1)."""
    return u**3 * (10 - 15*u + 6*u*u)

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
    """6x7 Jacobian for body ee_bid (stacked [Jv; Jw]) restricted to arm DOFs."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
    return np.vstack([jacp[:, daddr], jacr[:, daddr]])  # 6x7

def logdet_manip(J, reg=1e-8):
    """Return log det(J J^T + reg I)."""
    A = J @ J.T + reg * np.eye(J.shape[0])
    sign, logdet = np.linalg.slogdet(A)
    # If sign <= 0 due to numerical issues, heavily penalize
    return logdet if sign > 0 else -1e12

def grad_logdet_manip_fd(model, data_fd, qaddr, daddr, ee_bid, q_base, eps=1e-4, reg=1e-8):
    """
    Finite-difference gradient of logdet(JJ^T) at q_base.
    Uses a scratch MjData (data_fd) so we don't disturb the main sim.
    """
    n = len(qaddr)
    g = np.zeros(n)
    # Backup original
    qtmp = np.zeros_like(data_fd.qpos[qaddr])

    for i in range(n):
        # +eps
        data_fd.qpos[:] = 0.0
        data_fd.qpos[:] = data_fd.qpos  # no-op to keep shape; we’ll set only the arm joints next
        # set to base
        data_fd.qpos[qaddr] = q_base
        data_fd.qpos[qaddr[i]] = q_base[i] + eps
        mujoco.mj_forward(model, data_fd)
        Jp = compute_body_J(model, data_fd, ee_bid, daddr)
        f_plus = logdet_manip(Jp, reg=reg)

        # -eps
        data_fd.qpos[qaddr] = q_base
        data_fd.qpos[qaddr[i]] = q_base[i] - eps
        mujoco.mj_forward(model, data_fd)
        Jm = compute_body_J(model, data_fd, ee_bid, daddr)
        f_minus = logdet_manip(Jm, reg=reg)

        g[i] = (f_plus - f_minus) / (2.0 * eps)

    # restore (not strictly needed since we never step data_fd)
    data_fd.qpos[qaddr] = q_base
    mujoco.mj_forward(model, data_fd)
    return g

def nullspace_projector(J, J_pinv):
    n = J_pinv.shape[0]
    return np.eye(n) - J_pinv @ J


# =========================
# Model setup
# =========================
def load_model_and_indices():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Reset to "home" keyframe if present
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
# Main loop (IK + trajectory + manipulability null-space)
# =========================
def run():
    model, data, joint_ids, qaddr, daddr, ee_bid = load_model_and_indices()

    # Scratch data for finite-difference Jacobians (reused every tick)
    data_fd = mujoco.MjData(model)

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

    # Keep gripper open if present
    if model.nu >= 8:
        data.ctrl[7] = GRIPPER_OPEN

    # Timing
    t0 = data.time
    printed_converged = False
    printed_traj_done = False
    tick = 0
    last_grad = np.zeros_like(q_ref)

    print("\n=== Pose IK (trajectory -> hold) with Null-space Manipulability Maximization ===")
    print(f"Start position: {p_start}, Start quat (wxyz): {q_start}")
    print(f"Goal  position: {p_goal},  Goal  quat (wxyz): {q_goal}")
    print(f"TRAJ_TIME = {TRAJ_TIME}s, easing = {'minjerk' if USE_MINJERK else 'linear'}")
    print(f"NS_GAIN = {NS_GAIN}, FD_eps = {NS_EPS_FD}, reg = {MANIP_LOGDET_REG}, compute every {NS_EVERY} ticks\n")

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            # Trajectory progress 0..1
            u = (data.time - t0) / TRAJ_TIME
            if u < 0.0: u = 0.0
            if u > 1.0: u = 1.0
            s = minjerk(u) if USE_MINJERK else u

            # Interpolated desired pose in WORLD
            p_des = (1.0 - s) * p_start + s * p_goal
            q_des = slerp(q_start, q_goal, s)

            if (not printed_traj_done) and (u >= 1.0):
                printed_traj_done = True
                print("[Trajectory complete] Now holding final pose.")

            # Current EE pose
            p_cur, q_cur = get_body_pose_world(data, ee_bid)

            # Errors (world frame)
            e_pos = (p_des - p_cur)
            e_ang = quat_log_error(q_cur, q_des)

            pos_err_norm = np.linalg.norm(e_pos)
            ang_err_norm = np.linalg.norm(e_ang)

            if (not printed_converged) and (pos_err_norm <= POS_TOL) and (ang_err_norm <= ANG_TOL):
                printed_converged = True
                print(f"[Converged] |e_pos|={pos_err_norm:.4e}, |e_ang|={ang_err_norm:.4e}")

            # Proportional twist with clamping
            v_des = K_POS * e_pos
            w_des = K_ANG * e_ang
            v_norm = np.linalg.norm(v_des)
            if v_norm > VEL_LIN_MAX and v_norm > 0.0:
                v_des *= (VEL_LIN_MAX / v_norm)
            w_norm = np.linalg.norm(w_des)
            if w_norm > VEL_ANG_MAX and w_norm > 0.0:
                w_des *= (VEL_ANG_MAX / w_norm)
            xd = np.hstack([v_des, w_des])  # (6,)

            # Jacobian 6x7 (body "hand")
            J = compute_body_J(model, data, ee_bid, daddr)

            # Task-space DLS IK
            J_pinv = dls_pseudoinverse(J, LAMBDA)            # 7x6
            qdot_task = J_pinv @ xd                           # (7,)

            # Null-space manipulability ascent (compute every NS_EVERY ticks)
            if tick % NS_EVERY == 0:
                last_grad = grad_logdet_manip_fd(
                    model, data_fd, qaddr, daddr, ee_bid,
                    q_base=q_ref, eps=NS_EPS_FD, reg=MANIP_LOGDET_REG
                )
            qdot_ns_des = NS_GAIN * last_grad
            if NS_CLAMP is not None:
                nrm = np.linalg.norm(qdot_ns_des)
                if nrm > NS_CLAMP and nrm > 0:
                    qdot_ns_des *= NS_CLAMP / nrm

            N = nullspace_projector(J, J_pinv)                # 7x7
            qdot = qdot_task + N @ qdot_ns_des

            # Integrate, clamp, and command
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8:
                data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
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
    print("Viewer closed. Exiting.")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    run()
