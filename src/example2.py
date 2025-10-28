#!/usr/bin/env python3
import os, sys, mujoco, time
from mujoco import viewer
import numpy as np

# add repo root for imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from utils.ik.ik_qp import IKQPController

XML = "mujoco_menagerie/franka_emika_panda/scene.xml"

def axis_angle_to_quat(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle_rad
    return np.array([np.cos(half), *(np.sin(half) * axis)])

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

# reset to keyframe 'home' if present
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid != -1:
    mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)

ik = IKQPController(
    m, d,
    arm_joints=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"],
    ee_body="hand",
    task_mode="pose6",          # try "pos3" if you want pos-only
    null_objective="none",  # 'manigrad' | 'posture' | 'none'
    beta=0.25,
    qdot_max=1.2,
)

# ----- define a specific world-frame target pose -----
# start from current hand pose
hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
p_now = d.xpos[hand_bid].copy()
q_now = d.xquat[hand_bid].copy()  # w, x, y, z

# move +10 cm in X and +5 cm in Z (world frame)
p_goal = p_now + np.array([0.10, 0.0, 0.05])

# rotate 15° about world Z (yaw). Compose with current orientation.
q_yaw15 = axis_angle_to_quat([1, 0, 0], np.deg2rad(0))
# Final target orientation = q_now * q_yaw15
# (using the same Hamilton product convention as in IKQPController)
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
q_goal = quat_mul(q_now, q_yaw15)

# set the explicit pose target
ik.set_target(p_goal, q_goal)
# -----------------------------------------------

with viewer.launch_passive(m, d) as v:
    t0 = time.time()
    while v.is_running():
        qdot = ik.step()
        v.sync()
