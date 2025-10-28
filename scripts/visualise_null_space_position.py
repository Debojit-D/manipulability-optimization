#!/usr/bin/env python3
"""
Null-space motion demo for your Panda XML (no site — uses the 'hand' body).
- Holds the current hand pose (position + orientation) with a task-space stabilizer.
- Injects joint motion in the Jacobian null space so the EE pose stays approximately fixed.
- Works with your <general> position actuators (actuator1..actuator7) and tendon actuator (actuator8).
"""

import time
import numpy as np
import mujoco
from mujoco import viewer

# ----------------- user knobs -----------------
XML_PATH = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/mujoco_menagerie/franka_emika_panda/scene.xml"
EE_BODY_NAME = "hand"          # from your XML
ARM_JOINTS = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
DT = 1.0/300.0

# Task-space gains (small to just "hold" pose)
K_POS = 0.01
K_ORI = 0.01
LAMBDA = 1e-2      # damp for pseudoinverse
ALPHA = 0.1       # null-space drift scale
Z_MODE = "sinusoid"  # "sinusoid" | "constant"
# ----------------------------------------------


def quat_mul(q1, q2):
    """Hamilton product (w, x, y, z)."""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conj(q):
    w,x,y,z = q
    return np.array([w,-x,-y,-z])


def quat_to_rotvec(q):
    """Map unit quaternion to so(3) vector via log map (small-angle safe)."""
    q = q / np.linalg.norm(q)
    w, v = q[0], q[1:]
    nv = np.linalg.norm(v)
    w = np.clip(w, -1.0, 1.0)
    if nv < 1e-12:
        return np.zeros(3)
    angle = 2.0*np.arctan2(nv, w)
    if angle < 1e-12:
        return np.zeros(3)
    axis = v / nv
    return angle * axis


def clamp_to_joint_limits(q, model, joint_ids):
    """Clamp a 7-vector q to the MuJoCo joint ranges."""
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        # Some ranges (e.g., joint4) are both negative; still valid.
        out[i] = np.clip(out[i], min(lo, hi), max(lo, hi))
    return out


# Load model & data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# Reset to "home" keyframe if present (your XML defines it)
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

# Map joints → addresses
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in ARM_JOINTS]
qaddr = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
daddr = np.array([model.jnt_dofadr[jid]  for jid in joint_ids], dtype=int)

# EE body id
ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

# Target pose = current pose
p0 = data.xpos[ee_bid].copy()       # world position of body frame
q0 = data.xquat[ee_bid].copy()      # (w,x,y,z) orientation of body frame in world

# Start with current arm configuration as reference
q_ref = data.qpos[qaddr].copy()

# Keep gripper open (per your keyframe ctrl: 255)
# actuator layout in XML: 1..7 joints, 8th is tendon "split"
GRIPPER_CTRL_VALUE = 255.0
if model.nu >= 8:
    data.ctrl[7] = GRIPPER_CTRL_VALUE  # index 7 (0-based) → actuator8

with viewer.launch_passive(model, data) as v:
    t0 = time.time()
    while v.is_running():
        # Forward to update kinematics
        mujoco.mj_forward(model, data)

        # Current EE pose (body frame)
        p = data.xpos[ee_bid].copy()
        q = data.xquat[ee_bid].copy()

        # Body Jacobians (3 x nv each)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)

        # Take columns for our 7 arm dofs
        Jp = jacp[:, daddr]          # 3x7
        Jr = jacr[:, daddr]          # 3x7
        J  = np.vstack([Jp, Jr])     # 6x7

        # Task-space error (position + orientation)
        e_pos = p0 - p
        # orientation error: q_err = q_current^{-1} * q_target (body-frame error)
        q_err = quat_mul(quat_conj(q), q0)
        e_ori = quat_to_rotvec(q_err)
        e = np.hstack([K_POS*e_pos, K_ORI*e_ori])   # 6,

        # Damped pseudoinverse
        JJt = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJt + (LAMBDA**2)*np.eye(6))   # 7x6

        # Task joint velocity to hold pose
        qdot_task = J_pinv @ e

        # Null projector
        N = np.eye(7) - J_pinv @ J

        # Null-space drift z
        t = time.time() - t0
        if Z_MODE == "sinusoid":
            z = np.array([
                np.cos(0.1*t), np.sin(0.2*t), np.cos(0.2*t),
                0.0, np.sin(0.2*t), np.sin(0.2*t), np.cos(0.2*t)
            ])
        else:
            z = np.array([0,0,1,0,0,1,0], dtype=float)

        qdot = qdot_task + ALPHA*(N @ z)

        # Integrate desired joint positions and clamp to limits
        q_ref = q_ref + DT*qdot
        q_ref = clamp_to_joint_limits(q_ref, model, joint_ids)

        # Send desired positions to the first 7 actuators (position servos)
        # Assumes actuator1..actuator7 map to joint1..joint7 in order (true in your XML)
        data.ctrl[0:7] = q_ref

        # Keep gripper command persistent
        if model.nu >= 8:
            data.ctrl[7] = GRIPPER_CTRL_VALUE

        # Step sim
        mujoco.mj_step(model, data)
        v.sync()
