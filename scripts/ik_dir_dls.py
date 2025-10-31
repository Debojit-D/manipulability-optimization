#!/usr/bin/env python3
"""
Pose IK with DLS (world-frame target) — holds target indefinitely.
- Drives EE "hand" body to a target position and orientation (world frame).
- Uses 6-DoF twist servo: xd = [vx, vy, vz, wx, wy, wz]
- Orientation error uses quaternion log map (robust small/large-angle).

No teleop. The script does NOT auto-stop on convergence; it keeps holding
the target until you close the viewer window.
"""

import numpy as np
import mujoco
from mujoco import viewer

# =========================
# User knobs
# =========================
XML_PATH      = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
ARM_JOINTS    = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
EE_BODY_NAME  = "hand"                # end-effector body name in MJCF

DT            = 1.0/300.0             # control dt (s)
LAMBDA        = 1e-2                  # DLS damping (0.01 .. 0.1 usually fine)
K_POS         = 2.0                   # m/s per m position error
K_ANG         = 2.0                   # rad/s per rad orientation error
VEL_LIN_MAX   = 0.25                  # m/s clamp
VEL_ANG_MAX   = 2.0                   # rad/s clamp

POS_TOL       = 1e-3                  # m
ANG_TOL       = 1e-2                  # rad (angle-axis norm)

GRIPPER_OPEN  = 255.0                 # keep tendon/gripper open if present (optional)

# ---- Target Pose (WORLD frame) ----
# Option A: set with RPY degrees (ZYX convention: Rz(yaw)*Ry(pitch)*Rx(roll))
TARGET_POS_W    = np.array([0.25, 0.40, 0.45])      # meters (x,y,z in world)
TARGET_RPY_DEG  = (0.0, 20.0, 0.0)                  # roll, pitch, yaw in degrees

# Option B: or set quaternion directly (w, x, y, z). If not None, overrides RPY.
TARGET_QUAT_W   = None
# Example: TARGET_QUAT_W = np.array([0.7071, 0.0, 0.0, 0.7071])  # 90 deg about z


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
    """Right DLS pseudoinverse for fat J (6x7): J^T (J J^T + lam^2 I)^-1."""
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + (lam**2) * np.eye(J.shape[0]))

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    """(w,x,y,z)*(w,x,y,z)"""
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
    """RPY (rad) -> quaternion (w,x,y,z) for Rz(yaw)*Ry(pitch)*Rx(roll)."""
    cr = np.cos(roll*0.5);  sr = np.sin(roll*0.5)
    cp = np.cos(pitch*0.5); sp = np.sin(pitch*0.5)
    cy = np.cos(yaw*0.5);   sy = np.sin(yaw*0.5)
    # Z * Y * X
    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = cy*sp*cr + sy*cp*sr
    z = sy*cp*cr - cy*sp*sr
    return quat_normalize(np.array([w,x,y,z]))

def mat_to_quat(R):
    """Rotation matrix (3x3) -> quaternion (w,x,y,z)."""
    R = np.asarray(R).reshape(3,3)
    t = np.trace(R)
    if t > 0.0:
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
    """
    Quaternion orientation error (vector in R^3) via log map.
    Returns angle-axis vector: angle * axis (rad).
    We compute q_err = q_des * conj(q_cur), enforce hemisphere (w>=0).
    """
    q_cur = quat_normalize(q_cur)
    q_des = quat_normalize(q_des)
    q_err = quat_mul(q_des, quat_conj(q_cur))
    if q_err[0] < 0.0:
        q_err = -q_err
    w, x, y, z = q_err
    v = np.array([x, y, z])
    s = np.linalg.norm(v)
    if s < 1e-8:  # small-angle
        return 2.0 * v
    angle = 2.0 * np.arctan2(s, w)
    axis  = v / s
    return angle * axis

def get_body_pose_world(data, bid):
    """Returns (pos_W, quat_W) of a body (world frame)."""
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
# Main loop (IK servo)
# =========================
def run():
    model, data, joint_ids, qaddr, daddr, ee_bid = load_model_and_indices()

    # Choose target quaternion
    if TARGET_QUAT_W is None:
        roll, pitch, yaw = np.deg2rad(TARGET_RPY_DEG)
        q_W_des = euler_zyx_to_quat(roll, pitch, yaw)
    else:
        q_W_des = quat_normalize(TARGET_QUAT_W)
    p_W_des = TARGET_POS_W.astype(float).copy()

    # Initial joint ref from current state
    q_ref = data.qpos[qaddr].copy()

    # Keep gripper open if present
    if model.nu >= 8:
        data.ctrl[7] = GRIPPER_OPEN

    print("\n=== Pose IK (world-frame target — HOLD MODE) ===")
    print(f"Target position (m): {p_W_des}")
    print(f"Target quaternion (w,x,y,z): {q_W_des}\n")

    printed_converged = False

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)

            # Current EE pose in world
            p_W_cur, q_W_cur = get_body_pose_world(data, ee_bid)

            # Errors (world frame)
            e_pos = (p_W_des - p_W_cur)               # m
            e_ang = quat_log_error(q_W_cur, q_W_des)  # rad-axis (3,)

            pos_err_norm = np.linalg.norm(e_pos)
            ang_err_norm = np.linalg.norm(e_ang)

            # Log convergence once (no stopping)
            if not printed_converged and (pos_err_norm <= POS_TOL) and (ang_err_norm <= ANG_TOL):
                printed_converged = True
                print(f"[Converged] |e_pos|={pos_err_norm:.4e}, |e_ang|={ang_err_norm:.4e}")

            # Desired twist (world): proportional servo with per-norm clamping
            v_des = K_POS * e_pos
            w_des = K_ANG * e_ang
            # clamp norms
            v_norm = np.linalg.norm(v_des)
            if v_norm > VEL_LIN_MAX and v_norm > 0.0:
                v_des *= (VEL_LIN_MAX / v_norm)
            w_norm = np.linalg.norm(w_des)
            if w_norm > VEL_ANG_MAX and w_norm > 0.0:
                w_des *= (VEL_ANG_MAX / w_norm)

            xd = np.hstack([v_des, w_des])  # (6,)

            # EE Jacobian at body "hand": 6x7
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
            J = np.vstack([jacp[:, daddr], jacr[:, daddr]])  # 6x7

            # DLS IK: qdot = J^† xd
            J_pinv = dls_pseudoinverse(J, LAMBDA)            # 7x6
            qdot   = J_pinv @ xd                              # (7,)

            # Integrate & clamp, then command position actuators
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8:
                data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            v.sync()

    # Final report after you close the viewer
    mujoco.mj_forward(model, data)
    p_W_cur, q_W_cur = get_body_pose_world(data, ee_bid)
    e_pos = p_W_des - p_W_cur
    e_ang = quat_log_error(q_W_cur, q_W_des)
    print("\nFinal pose error:")
    print(f"  |pos| = {np.linalg.norm(e_pos):.6f} m,  e_pos = {e_pos}")
    print(f"  |ang| = {np.linalg.norm(e_ang):.6f} rad, e_ang = {e_ang}")
    print("Viewer closed. Exiting.")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    run()
