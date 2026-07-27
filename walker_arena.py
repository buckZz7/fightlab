"""WalkerArena: 2-bot combat arena using the Lucky Robots walker policy.

Uses G1Controller from run.py directly — proven to maintain balance.
"""
import os, sys, json, math
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Load run.py code (G1Controller is proven to work)
_env = {"__file__": os.path.join(SCRIPT_DIR, "run.py"), "__name__": "__main__"}
with open(os.path.join(SCRIPT_DIR, "run.py")) as _f:
    exec(_f.read().split("def main")[0], _env)
ONNXPolicy = _env["ONNXPolicy"]
G1Controller = _env["G1Controller"]


def create_walker_controller(model, data):
    """Create a G1Controller with walker policy loaded."""
    walker = ONNXPolicy(os.path.join(SCRIPT_DIR, "walker.onnx"))
    croucher = ONNXPolicy(os.path.join(SCRIPT_DIR, "croucher.onnx"))
    rotator = ONNXPolicy(os.path.join(SCRIPT_DIR, "rotator.onnx"))
    try:
        reacher = ONNXPolicy(os.path.join(SCRIPT_DIR, "right_reacher.onnx"))
    except:
        reacher = None
    cfg = json.load(open(os.path.join(SCRIPT_DIR, "model_config.json")))
    ctrl = G1Controller(model, data, walker, croucher, rotator, cfg, reacher)
    ctrl._cache_actuator_ids()
    return ctrl


def init_robot(controller):
    """Set robot to default standing pose."""
    d = controller.data
    m = controller.model
    for i, name in enumerate(controller.joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = controller.default_joint_pos[i]
    mujoco.mj_forward(m, d)


def step_walker(controller, vel_cmd, arm_targets=None):
    """Step the walker. Returns 29 joint targets.

    vel_cmd: [vx, vy, yaw_rate] velocity command
    arm_targets: optional 14 values to override arm joints (indices 15-28)

    The walker maintains balance. Arm override for punching.
    """
    # Set velocity command
    controller.lin_vel_x = float(vel_cmd[0])
    controller.lin_vel_y = float(vel_cmd[1])
    controller.ang_vel_z = float(vel_cmd[2])

    # Get walker targets (G1Controller.step handles obs + inference)
    targets = controller.step()

    # Override arm joints if provided
    if arm_targets is not None:
        arm_indices = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        for i, idx in enumerate(arm_indices):
            targets[idx] = arm_targets[i]

    # Apply to actuators
    for i, aid in enumerate(controller.actuator_ids):
        if aid >= 0:
            controller.data.ctrl[aid] = targets[i]

    return targets


def get_pelvis_z(controller):
    p = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return float(controller.data.xpos[p][2])
