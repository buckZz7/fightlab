"""G1 balance substrate for Track B.

CRITICAL (verified 2026-07): the unitree_rl_mjlab velocity ONNX
(LocoBase29) FALLS in 2.8s-13.4s even in its NATIVE
scene_29dof.xml. It is NOT a usable balance base. Do not use it.

For Track B we use StandPD: pure PD-to-HOME on the legs/waist,
which is stable by construction (holds the known-good stand pose).
The SB3 fight policy then drives ARM residuals + walk cmd on top,
and learns balance maintenance via the env reward. This gives a
boxer that STANDS (no faceplant) and learns to PUNCH.
Footwork is a later warm-start extension.

Gains/SCALE/HOME from the unitree deploy config (deploy.yaml).
"""
import os
import numpy as np

# PD gains (per-joint, 29-DoF order matching the model):
#   legs 0-11 (hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll x2)
#   waist 12-14 (yaw,roll,pitch)
#   arms 15-28 (l_shoulder_p,r,y, l_elbow, l_wrist_r,p,y, r_shoulder..., r_elbow, r_wrist...)
# VALUES: documented stiff gains (lerobot/unitree-g1-mujoco README,
# NVIDIA GR00T spec). The OLD gains (40-99) were ~3x too soft and the
# G1 collapsed instantly. Hip/Knee/Ankle/Waist per GR00T; arms moderate.
KP = np.array([150, 150, 150, 300, 40, 40,        # L leg: hip,hip,hip,knee,ank,ank
               150, 150, 150, 300, 40, 40,        # R leg
               250, 250, 250,                      # waist yaw,roll,pitch
               100, 100, 100, 100, 100, 10, 10,    # L arm: sh_p,sh_r,sh_y,elbow,wrist_r,wrist_p,wrist_y
               100, 100, 100, 100, 100, 10, 10])   # R arm
KD = np.array([2, 2, 2, 4, 2, 2,
               2, 2, 2, 4, 2, 2,
               5, 5, 5,
               2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2])

# Standing pose (joint targets from neutral + slight knee/hip bend).
# Matches the unitree HOME used by the deploy policy.
HOME = np.array([-0.1, 0, 0, 0.3, -0.2, 0,
                 -0.1, 0, 0, 0.3, -0.2, 0,
                 0, 0, 0,
                 0.35, 0.18, 0, 0.87, 0, 0, 0,
                 0.35, -0.18, 0, 0.87, 0, 0, 0])

# Per-joint scale for residuals (how far the policy can push from HOME).
SCALE = np.array([0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                  0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                  0.55, 0.44, 0.44,
                  0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
                  0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07])


class StandPD:
    """Stable standing substrate: PD-to-HOME on all 29 joints.

    No ONNX. Holds the robot upright. The fight policy overrides
    the ARM portion (15:29) with learned residuals to punch, and
    can nudge the leg targets slightly for footwork.
    """
    def __init__(self):
        self.target = HOME.copy()
        self.cmd = np.zeros(3, dtype=np.float32)

    def set_command(self, vx=0.0, vy=0.0, wz=0.0):
        self.cmd[:] = [vx, vy, wz]

    def update(self, qpos, qvel, off=0):
        # StandPD does not change the base target from command; footwork
        # is applied by the policy via target_override in pd_torque.
        # (kept for interface compat with the env loop)
        return self.target

    def pd_torque(self, qpos, qvel, off=0, target_override=None):
        t = self.target if target_override is None else target_override
        return KP * (t - qpos[off+7:off+36]) - KD * qvel[off+6:off+35]

    def reset(self):
        self.target = HOME.copy()
        self.cmd[:] = 0


class LocoBase29:
    """DEPRECATED: unitree ONNX balance base. Verified UNSTABLE
    (falls 2.8s-13.4s). Kept only for reference / ablation.
    Use StandPD instead.
    """
    def __init__(self, onnx_path=None):
        raise RuntimeError(
            "LocoBase29 (ONNX) is UNSTABLE (falls 2.8-13.4s). "
            "Use StandPD from loco_base29 instead.")
