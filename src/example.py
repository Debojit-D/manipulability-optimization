import mujoco
from mujoco import viewer
# --- add this at the top of src/example.py ---
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
# ---------------------------------------------
from utils.nullspace_opt.nullspace_opt import NullSpaceOptimizer

from utils.nullspace_opt.nullspace_opt import NullSpaceOptimizer

XML = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/mujoco_menagerie/franka_emika_panda/scene.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

# reset to "home" keyframe if present
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid != -1: mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)

opt = NullSpaceOptimizer(
    m, d,
    arm_joints=["joint1","joint2","joint3","joint4","joint5","joint6","joint7"],
    ee_body="hand",
    task_mode="pose6",          # try "pos3" to see higher nullity
    null_policy="svd",     # 'manigrad' | 'svd' | 'random'
    alpha_null=0.04,
)

opt.set_target_from_current()

with viewer.launch_passive(m, d) as v:
    while v.is_running():
        opt.step()
        v.sync()
