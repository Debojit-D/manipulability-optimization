#!/usr/bin/env python3
"""
Move the end-effector to a TARGET POSITION using DLS IK (position-only).
- Target is given in the robot BASE frame (not relative to current EE pose).
- Orientation is held fixed (we only move position).
- No teleop / no keyboard input.
"""

import numpy as np
import mujoco
from mujoco import viewer

# =============== User knobs ===============
XML_PATH       = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
ARM_JOINTS     = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
EE_BODY_NAME   = "hand"

# >>> Set this to your robot's base body name in the MJCF <<<
# Common for Franka is "panda_link0". If unsure, set to None to treat WORLD as base.
BASE_BODY_NAME = "None"   # or None if your base is fixed at world origin

DT             = 1.0/300.0        # control dt (s)
K_POS          = 1.0              # task-space proportional gain (m/s per m error)
LAMBDA         = 1e-2             # DLS damping (0.01)
VEL_DAMP       = 0.05             # light joint-velocity damping
POS_TOL        = 1e-4             # stop when ||p_des - p|| < POS_TOL
MAX_STEPS      = 20000            # safety cap
GRIPPER_OPEN   = 255.0            # keep tendon/gripper open if present

# Target in BASE frame (meters). Example: +10 cm in X of base.
TARGET_POS_BASE = np.array([0.10, 0.00, 0.00], dtype=float)
# =========================================

# =============== Helpers ===============
def clamp_to_limits(q, model, joint_ids):
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        out[i] = np.clip(out[i], min(lo, hi), max(lo, hi))
    return out

def dls_pinv_pos(Jp, lam):
    """
    Position-only DLS pseudoinverse (3x7 Jp):
    Jp^T (Jp Jp^T + lam^2 I_3)^-1
    """
    JJt = Jp @ Jp.T          # 3x3
    return Jp.T @ np.linalg.inv(JJt + (lam**2) * np.eye(3))  # 7x3

def quat_to_mat(qwxyz):
    """Convert MuJoCo xquat (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = qwxyz
    # standard quaternion to rotation
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    R = np.array([
        [ww+xx-yy-zz,     2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z),   ww-xx+yy-zz,   2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x), ww-xx-yy+zz  ]
    ])
    return R

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
    ee_bid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

    if BASE_BODY_NAME is None:
        base_bid = -1  # use WORLD as base
    else:
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
        if base_bid == -1:
            raise RuntimeError(f"Base body '{BASE_BODY_NAME}' not found in model.")

    return model, data, joint_ids, qaddr, daddr, ee_bid, base_bid

# =============== Main ===============
def run():
    model, data, joint_ids, qaddr, daddr, ee_bid, base_bid = load_model_and_indices()
    q_ref = data.qpos[qaddr].copy()

    # Compute desired position in WORLD frame from a base-frame target
    if base_bid == -1:
        # Base is WORLD: world == base
        p_des_world = TARGET_POS_BASE.copy()
    else:
        # Base pose in WORLD
        p_base_w   = data.xpos[base_bid].copy()    # 3,
        q_base_w   = data.xquat[base_bid].copy()   # (w,x,y,z)
        R_base_w   = quat_to_mat(q_base_w)         # 3x3
        # Transform base-frame target into WORLD frame
        p_des_world = p_base_w + R_base_w @ TARGET_POS_BASE

    # Keep gripper open if present
    if model.nu >= 8:
        data.ctrl[7] = GRIPPER_OPEN

    print("\nTARGET (base frame):", TARGET_POS_BASE)
    print("TARGET (world frame):", p_des_world)

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        for _ in range(MAX_STEPS):
            mujoco.mj_forward(model, data)

            # Current EE position in WORLD frame
            p_w = data.xpos[ee_bid].copy()         # 3,
            pos_err_w = p_des_world - p_w
            if np.linalg.norm(pos_err_w) < POS_TOL:
                # Hold final posture when converged
                data.ctrl[0:7] = q_ref
                if model.nu >= 8:
                    data.ctrl[7] = GRIPPER_OPEN
                mujoco.mj_step(model, data)
                v.sync()
                continue

            # Desired translational velocity in WORLD (no orientation control)
            v_des_w = K_POS * pos_err_w            # 3,

            # Position Jacobian in WORLD (3 x nv) -> select 7 arm DOFs => 3x7
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))         # allocate even if unused (safer API usage)
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
            Jp = jacp[:, daddr]                    # 3x7

            # DLS IK (position-only) -> joint velocities
            Jp_pinv = dls_pinv_pos(Jp, LAMBDA)     # 7x3
            qdot = Jp_pinv @ v_des_w               # 7,

            # Light velocity damping (optional)
            qdot -= VEL_DAMP * data.qvel[daddr]

            # Integrate, clamp, and command
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8:
                data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            v.sync()

    print("Done.")

if __name__ == "__main__":
    run()
