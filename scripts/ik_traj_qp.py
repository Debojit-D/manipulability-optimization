#!/usr/bin/env python3
"""
Pose IK with QP (world-frame target) — trajectory to target + hold.
- Interpolates from current EE pose to target (world) over TRAJ_TIME seconds.
- Position uses linear interpolation; orientation uses SLERP.
- After reaching the target, continues to hold it indefinitely.
- Velocity IK solved as a QP with joint-limit (next-step) constraints.

Requires: cvxpy (uses OSQP by default).
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

# QP weights:  minimize  w_pos*||Jv - xd_pos||^2 + w_ang*||Jw - xd_ang||^2 + R*||qdot||^2
W_POS         = 1.0                   # weight for position tracking residual
W_ANG         = 1.0                   # weight for orientation tracking residual
QP_REG        = 1e-4                  # small Tikhonov regularization on qdot

POS_TOL       = 1e-3                  # m
ANG_TOL       = 1e-2                  # rad (angle-axis norm)

GRIPPER_OPEN  = 255.0                 # keep tendon/gripper open if present

# ---- Trajectory knobs ----
TRAJ_TIME       = 10.0                 # seconds to go from start -> goal
USE_MINJERK     = True                # smooth easing (5th-order); else linear

# ---- Target Pose (WORLD frame) ----
TARGET_POS_W    = np.array([0.25, 0.40, 0.45])      # meters (x,y,z in world)
TARGET_RPY_DEG  = (0.0, 0.0, 0.0)                   # roll, pitch, yaw in degrees
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
# QP builder
# =========================
def build_qp(n, dt, q_lo, q_hi):
    """
    Build a cvxpy problem:
        minimize  w_pos*|| (Jv - xd)[0:3] ||^2 + w_ang*|| (Jv - xd)[3:6] ||^2 + reg*||v||^2
        subject to  q + dt*v in [q_lo, q_hi]   (added per-step as parameters)
    Returns (problem, params) where params is a dict of cvxpy Parameters to update each tick.
    """
    v = cp.Variable(n)                                 # qdot
    Jp = cp.Parameter((6, n))                          # Jacobian
    xdp = cp.Parameter(6)                              # desired twist
    q_now = cp.Parameter(n)                            # current joint pos
    qlo = cp.Parameter(n)
    qhi = cp.Parameter(n)

    err = Jp @ v - xdp
    obj = (W_POS * cp.sum_squares(err[:3]) +
           W_ANG * cp.sum_squares(err[3:]) +
           QP_REG * cp.sum_squares(v))

    cons = [
        qlo <= q_now + dt * v,
        q_now + dt * v <= qhi
    ]

    prob = cp.Problem(cp.Minimize(obj), cons)

    params = dict(v=v, Jp=Jp, xdp=xdp, q_now=q_now, qlo=qlo, qhi=qhi)
    return prob, params


# =========================
# Main loop (QP IK + trajectory)
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

    # Joint limits vectors
    q_lo = np.array([model.jnt_range[jid][0] for jid in joint_ids])
    q_hi = np.array([model.jnt_range[jid][1] for jid in joint_ids])

    # QP setup (reuse the same problem each tick via Parameters)
    prob, prm = build_qp(n, DT, q_lo, q_hi)

    # Keep gripper open if present
    if model.nu >= 8:
        data.ctrl[7] = GRIPPER_OPEN

    # Timing
    t0 = data.time
    printed_converged = False
    printed_traj_done = False

    print("\n=== Pose IK (QP trajectory -> hold) ===")
    print(f"Start position: {p_start}, Start quat (wxyz): {q_start}")
    print(f"Goal  position: {p_goal},  Goal  quat (wxyz): {q_goal}")
    print(f"TRAJ_TIME = {TRAJ_TIME}s, easing = {'minjerk' if USE_MINJERK else 'linear'}")
    print("QP solver: OSQP (fallback to ECOS if needed)\n")

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

            # Task-space errors (world frame)
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
            # clamp task-space rates
            v_norm = np.linalg.norm(v_des)
            if v_norm > VEL_LIN_MAX and v_norm > 0.0:
                v_des *= (VEL_LIN_MAX / v_norm)
            w_norm = np.linalg.norm(w_des)
            if w_norm > VEL_ANG_MAX and w_norm > 0.0:
                w_des *= (VEL_ANG_MAX / w_norm)
            xd = np.hstack([v_des, w_des])  # desired 6D twist

            # Jacobian 6x7 (body "hand") for arm dofs only
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
            J = np.vstack([jacp[:, daddr], jacr[:, daddr]])  # 6x7

            # Update QP parameters
            prm["Jp"].value = J
            prm["xdp"].value = xd
            prm["q_now"].value = q_ref
            prm["qlo"].value = q_lo
            prm["qhi"].value = q_hi

            # Solve QP
            try:
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    # Fallback
                    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
            except Exception:
                # As a last resort, try ECOS
                prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

            if prm["v"].value is None:
                # If the solver failed, fall back to zero motion this tick
                qdot = np.zeros(n)
            else:
                qdot = np.array(prm["v"].value).reshape(-1)

            # Integrate & clamp; command position actuators
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8:
                data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            v.sync()

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
