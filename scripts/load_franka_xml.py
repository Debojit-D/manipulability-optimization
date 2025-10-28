# ------------------------------------------------------------------------------
# Load a MuJoCo model and start the viewer at the "home" keyframe (if present).
# If "home" doesn't exist, it lists available keyframes and uses the default state.
# ------------------------------------------------------------------------------

import mujoco
import numpy as np
from mujoco.viewer import launch

# Path to your MuJoCo XML model (MJCF or converted URDF file)
XML_PATH = "/home/iitgn-robotics/Debojit_WS/manipulability-optimization/mujoco_menagerie/franka_emika_panda/scene.xml"

# Load model & data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Try resetting to the "home" keyframe
key_name = "home"
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)

if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
else:
    # List available keyframes to help you pick the right name
    if model.nkey > 0:
        print(f'[WARN] Keyframe "{key_name}" not found. Available keyframes:')
        for i in range(model.nkey):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            print(f"  - {i}: {nm}")
    else:
        print("[WARN] No keyframes defined in this model.")
    # Fall back to a clean reset (default initial state)
    mujoco.mj_resetData(model, data)

# (Optional) do one step to initialize derived quantities
mujoco.mj_forward(model, data)

# Launch viewer
viewer = launch(model, data)

# Render loop (no simulation stepping; pure visualization)
while viewer and viewer.is_running():
    viewer.render()
