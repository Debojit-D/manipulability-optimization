#!/usr/bin/env python3
"""
Minimal joint-velocity teleop for 7-DoF arm.
Keys: q..u (+) and a..j (-) for joints 1..7, hold to move. ESC quits, 'z' zeros.
"""

import numpy as np
import mujoco
from mujoco import viewer
from pynput import keyboard

# ---------- user knobs ----------
XML_PATH = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
ARM_JOINTS = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
DT        = 1.0/300.0          # control dt (s)
SPEED     = 0.2                # rad/s when key held
VEL_DAMP  = 0.05               # light damping on joint velocities
GRIPPER_OPEN = 255.0           # keep tendon/gripper open if present

PLUS_KEYS  = ['q','w','e','r','t','y','u']
MINUS_KEYS = ['a','s','d','f','g','h','j']
# --------------------------------

def clamp_to_limits(q, model, joint_ids):
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        out[i] = np.clip(out[i], min(lo, hi), max(lo, hi))
    return out

# Load model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# Reset (use 'home' keyframe if exists)
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1: mujoco.mj_resetDataKeyframe(model, data, key_id)
else:            mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Joint indices
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
qaddr     = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
daddr     = np.array([model.jnt_dofadr[jid]  for jid in joint_ids], dtype=int)

# Reference joint positions (we'll integrate velocities into this and command positions)
q_ref = data.qpos[qaddr].copy()

# Keep gripper open (if actuator8 exists)
if model.nu >= 8:
    data.ctrl[7] = GRIPPER_OPEN

# Keyboard handling
keys_down = set()
stop_flag = False

def on_press(key):
    global stop_flag
    try:
        k = key.char
    except AttributeError:
        if key == keyboard.Key.esc:
            stop_flag = True
            return False
        return
    if k == 'z':
        keys_down.clear()
        return
    keys_down.add(k)

def on_release(key):
    try:
        k = key.char
    except AttributeError:
        return
    keys_down.discard(k)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("\n=== Joint-velocity teleop ===")
print("Hold keys q..u = + for joints 1..7, a..j = - for joints 1..7")
print("Press 'z' to zero, ESC to quit.\n")

with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
    while v.is_running() and not stop_flag:
        mujoco.mj_forward(model, data)

        # Build joint velocity command from held keys
        qdot_cmd = np.zeros(7)
        for i, k in enumerate(PLUS_KEYS):
            if k in keys_down: qdot_cmd[i] += SPEED
        for i, k in enumerate(MINUS_KEYS):
            if k in keys_down: qdot_cmd[i] -= SPEED

        # Light damping to reduce residual oscillations
        qdot_cmd -= VEL_DAMP * data.qvel[daddr]

        # Integrate and clamp
        q_ref = clamp_to_limits(q_ref + DT * qdot_cmd, model, joint_ids)

        # Send to position actuators (assumes actuator1..7 map to joints 1..7)
        data.ctrl[0:7] = q_ref
        if model.nu >= 8:
            data.ctrl[7] = GRIPPER_OPEN

        mujoco.mj_step(model, data)
        v.sync()

listener.stop()
print("Exiting teleop.")
