#!/usr/bin/env python3
"""
Null-space motion demo with real-time keyboard control (hold-to-move).

- Holds current end-effector pose.
- Allows manual null-space motion with keys q..u (+) and a..j (-).
- Motion only occurs while keys are pressed.
"""

import time
import numpy as np
import mujoco
from mujoco import viewer
from pynput import keyboard


# ==========================================================
# ----------------- USER PARAMETERS -------------------------
# ==========================================================
XML_PATH = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/robot_description/franka_emika_panda/scene.xml"
EE_BODY_NAME = "hand"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

DT = 1.0 / 100.0
K_POS = 0.005
K_ORI = 0.005
LAMBDA = 5e-2
ALPHA = 0.1
STEP = 1.0  # speed multiplier for null motion

PLUS_KEYS  = ['q', 'w', 'e', 'r', 't', 'y', 'u']
MINUS_KEYS = ['a', 's', 'd', 'f', 'g', 'h', 'j']
# ==========================================================


# ==========================================================
# ----------------- UTILITY FUNCTIONS -----------------------
# ==========================================================
def quat_mul(q1, q2):
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conj(q):
    """Quaternion conjugate."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_rotvec(q):
    """Map unit quaternion to so(3) rotation vector via logarithmic map."""
    q = q / np.linalg.norm(q)
    w, v = q[0], q[1:]
    nv = np.linalg.norm(v)
    w = np.clip(w, -1.0, 1.0)
    if nv < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(nv, w)
    return angle * v / nv


def clamp_to_joint_limits(q, model, joint_ids):
    """Clamp joint positions to their limits."""
    out = q.copy()
    for i, jid in enumerate(joint_ids):
        lo, hi = model.jnt_range[jid]
        out[i] = np.clip(out[i], min(lo, hi), max(lo, hi))
    return out
# ==========================================================


# ==========================================================
# ----------------- MODEL INITIALIZATION -------------------
# ==========================================================
def load_model():
    """Load MuJoCo model and initialize data."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)
    return model, data


def get_joint_indices(model):
    """Get joint indices, addresses, and end-effector body ID."""
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
    qaddr = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    daddr = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=int)
    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    return joint_ids, qaddr, daddr, ee_bid
# ==========================================================


# ==========================================================
# ----------------- KEYBOARD HANDLING -----------------------
# ==========================================================
class KeyboardController:
    """Handles keyboard input using pynput for real-time null-space control."""

    def __init__(self):
        self.keys_down = set()
        self.stop_flag = False
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            if key == keyboard.Key.esc:
                self.stop_flag = True
                return False
            return
        self.keys_down.add(k)

    def on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            return
        if k in self.keys_down:
            self.keys_down.remove(k)

    def compute_null_vector(self):
        """Return 7D null-space direction vector based on currently pressed keys."""
        z = np.zeros(7)
        for i, k in enumerate(PLUS_KEYS):
            if k in self.keys_down:
                z[i] += STEP
        for i, k in enumerate(MINUS_KEYS):
            if k in self.keys_down:
                z[i] -= STEP
        return z
# ==========================================================


# ==========================================================
# ----------------- MAIN CONTROL LOOP ----------------------
# ==========================================================
def run_nullspace_control(model, data, joint_ids, qaddr, daddr, ee_bid):
    """Main control loop for null-space motion with keyboard input."""
    # Initialize reference and target pose
    p0 = data.xpos[ee_bid].copy()
    q0 = data.xquat[ee_bid].copy()
    q_ref = data.qpos[qaddr].copy()

    if model.nu >= 8:
        data.ctrl[7] = 255.0  # Keep gripper open

    # Initialize keyboard handler
    kb = KeyboardController()

    print("\n=== Null-space keyboard control (hold keys to move) ===")
    print(" q..u : + joints 1–7 | a..j : - joints 1–7 | Esc : quit\n")

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        while v.is_running() and not kb.stop_flag:
            mujoco.mj_forward(model, data)

            # Null-space direction based on current key states
            z = kb.compute_null_vector()

            # Get current EE pose
            p = data.xpos[ee_bid].copy()
            q = data.xquat[ee_bid].copy()

            # Compute Jacobians
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
            J = np.vstack([jacp[:, daddr], jacr[:, daddr]])

            # Task-space stabilization
            e_pos = p0 - p
            q_err = quat_mul(quat_conj(q), q0)
            e_ori = quat_to_rotvec(q_err)
            e = np.hstack([K_POS * e_pos, K_ORI * e_ori])

            JJt = J @ J.T
            J_pinv = J.T @ np.linalg.inv(JJt + (LAMBDA**2) * np.eye(6))
            qdot_task = J_pinv @ e
            N = np.eye(7) - J_pinv @ J

            VEL_DAMP = 0.05  # put this near your other knobs at the top
            qdot = qdot_task + ALPHA * (N @ z) - VEL_DAMP * data.qvel[daddr]
            
            q_ref = clamp_to_joint_limits(q_ref + DT * qdot, model, joint_ids)


            # Send to actuators
            data.ctrl[0:7] = q_ref
            if model.nu >= 8:
                data.ctrl[7] = 255.0

            mujoco.mj_step(model, data)
            v.sync()

    kb.listener.stop()
    print("Exiting cleanly.")
# ==========================================================


# ==========================================================
# ----------------- MAIN ENTRY POINT -----------------------
# ==========================================================
def main():
    model, data = load_model()
    joint_ids, qaddr, daddr, ee_bid = get_joint_indices(model)
    run_nullspace_control(model, data, joint_ids, qaddr, daddr, ee_bid)


if __name__ == "__main__":
    main()
