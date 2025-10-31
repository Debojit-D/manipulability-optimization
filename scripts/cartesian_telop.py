#!/usr/bin/env python3
"""
Minimal Cartesian teleop using DLS IK (6-DoF twist; moves only while keys are held).

Linear (m/s):   q/a = +x/−x,  w/s = +y/−y,  e/d = +z/−z
Angular (rad/s): r/f = +roll/−roll,  t/g = +pitch/−pitch,  y/h = +yaw/−yaw
ESC quits, 'z' zeros all commands.
"""

import numpy as np
import mujoco
from mujoco import viewer
from pynput import keyboard

# =========================
# User knobs
# =========================
XML_PATH      = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
ARM_JOINTS    = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
DT            = 1.0/300.0              # control dt (s)
LIN_SPEED     = 0.05                   # m/s per key
ANG_SPEED     = 0.5                    # rad/s per key
LAMBDA        = 0.1                   # DLS damping (0.01)
GRIPPER_OPEN  = 255.0                  # keep tendon/gripper open if present

# Key map: build 6D twist [vx, vy, vz, wx, wy, wz]
LIN_PLUS  = {'q':0, 'w':1, 'e':2}
LIN_MINUS = {'a':0, 's':1, 'd':2}
ANG_PLUS  = {'r':0, 't':1, 'y':2}      # roll, pitch, yaw +
ANG_MINUS = {'f':0, 'g':1, 'h':2}      # roll, pitch, yaw −


# =========================
# Utilities
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

def build_twist_from_keys(keys_down):
    """Return 6D twist [vx, vy, vz, wx, wy, wz] from held keys and a flag any_key."""
    xd = np.zeros(6)
    any_key = False
    for k, i in LIN_PLUS.items():
        if k in keys_down: xd[i] += LIN_SPEED; any_key = True
    for k, i in LIN_MINUS.items():
        if k in keys_down: xd[i] -= LIN_SPEED; any_key = True
    for k, i in ANG_PLUS.items():
        if k in keys_down: xd[3+i] += ANG_SPEED; any_key = True
    for k, i in ANG_MINUS.items():
        if k in keys_down: xd[3+i] -= ANG_SPEED; any_key = True
    return xd, any_key


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

    # Joint indices & addresses
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
    qaddr     = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    daddr     = np.array([model.jnt_dofadr[jid]  for jid in joint_ids], dtype=int)

    # End-effector body (hand)
    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    return model, data, joint_ids, qaddr, daddr, ee_bid


# =========================
# Keyboard handling
# =========================
class KeyState:
    def __init__(self):
        self.keys_down = set()
        self.stop = False
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            if key == keyboard.Key.esc:
                self.stop = True
                return False
            return
        if k == 'z':
            self.keys_down.clear()
            return
        self.keys_down.add(k)

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            return
        self.keys_down.discard(k)

    def stop_listener(self):
        self.listener.stop()


# =========================
# Main loop
# =========================
def run():
    model, data, joint_ids, qaddr, daddr, ee_bid = load_model_and_indices()
    q_ref = data.qpos[qaddr].copy()

    # Keep gripper open if present
    if model.nu >= 8:
        data.ctrl[7] = GRIPPER_OPEN

    keys = KeyState()

    print("\n=== Cartesian DLS-IK teleop (hold keys to move) ===")
    print("Linear:  q/a +x/−x, w/s +y/−y, e/d +z/−z")
    print("Angular: r/f +roll/−roll, t/g +pitch/−pitch, y/h +yaw/−yaw")
    print("Press 'z' to zero, ESC to quit.\n")

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running() and not keys.stop:
            mujoco.mj_forward(model, data)

            # Build desired twist from keys; if none pressed, do nothing
            xd, any_key = build_twist_from_keys(keys.keys_down)
            if not any_key:
                mujoco.mj_step(model, data)
                v.sync()
                continue

            # EE body Jacobians (3×nv each); select 7 arm DOFs -> 3×7 each, stack -> 6×7
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
            J = np.vstack([jacp[:, daddr], jacr[:, daddr]])  # 6x7

            # DLS IK: qdot = J^† xd
            J_pinv = dls_pseudoinverse(J, LAMBDA)            # 7x6
            qdot   = J_pinv @ xd                              # 7,

            # Integrate and clamp, then command position actuators
            q_ref = clamp_to_limits(q_ref + DT * qdot, model, joint_ids)
            data.ctrl[0:7] = q_ref
            if model.nu >= 8:
                data.ctrl[7] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            v.sync()

    keys.stop_listener()
    print("Exiting Cartesian teleop.")


# =========================
# Entry
# =========================
if __name__ == "__main__":
    run()
