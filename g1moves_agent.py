"""Track B hybrid fighter: balance base (legs) + G1 Moves strikes (arms).

Proven-stable combination:
  - LEGS + WAIST (joints 0-14): driven by LocoBase29 (the 29-DoF
    velocity ONNX). This is what kept Gen1-3 upright for 500 steps.
    It also does footwork via set_command(vx,vy,wz).
  - ARMS (joints 15-28): driven by G1 Moves strike blends (jab /
    low punch / rapid punch). Full-body prior gives real punch shape.

Result: a balanced, footwork-capable, full-body striker in OUR sim.
This is the Track B agent the fight layer rides on top of.

G1MovesAgent.step() returns a 29-torque vector combining both.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from loco_base29 import LocoBase29, HOME, KP, KD

# G1 Moves PD gains (same as loco; arms use these for strike blend)
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
STANCE_ARMS = np.zeros(14, dtype=np.float32)  # arm targets relative to HOME
# fighting guard: hands up (shoulder pitch ~0.4 from HOME)
STANCE_ARMS[0] = 0.4   # left shoulder pitch offset
STANCE_ARMS[6] = 0.4   # right shoulder pitch offset

SKILLS = ["jab", "lowpunch", "rapidpunch", "frontkick", "sidekick", "snapkick"]
SKILL_FILES = {
    "jab": "M_ShortMove12_quickjab",
    "lowpunch": "M_Move2_lowpunch",
    "rapidpunch": "M_Move7_rapidpunch",
    "frontkick": "M_Move18_frontkick",
    "sidekick": "M_Move10_sidekick",
    "snapkick": "M_ShortMove13_snapkick",
}
ARMS = slice(15, 29)      # arm joints in 29-DoF
LEGS_WAIST = slice(0, 15)  # legs + waist


class G1MovesAgent:
    def __init__(self, motion_dir, onnx_path=None, blend_rate=0.3, control_dt=0.02):
        self.motion_dir = motion_dir
        self.blend = blend_rate
        self.control_dt = control_dt
        self.loco = LocoBase29(onnx_path) if onnx_path else LocoBase29()
        # load arm clips
        self.clips = {}
        for sk, fn in SKILL_FILES.items():
            p = os.path.join(motion_dir, "motion", f"{fn}.npz")
            if os.path.exists(p):
                m = np.load(p)
                # arm portion of joint_pos (15:29)
                self.clips[sk] = {
                    "arm": m["joint_pos"].astype(np.float32)[:, ARMS],
                    "fps": float(m["fps"]),
                }
        self.active = None
        self.active_t = 0.0
        self.arm_target = np.zeros(14, dtype=np.float32)  # relative to HOME arms

    def command(self, skill_id, walk_cmd=(0, 0, 0)):
        """skill_id: int [0..len(SKILLS)-1], -1 = stance. walk_cmd=(vx,vy,wz)."""
        self.loco.set_command(*walk_cmd)
        if skill_id == -1 or skill_id >= len(SKILLS):
            self.active = None
        else:
            name = SKILLS[skill_id]
            if self.active != name:
                self.active = name
                self.active_t = 0.0

    def step(self, model, data, qpos_off=0, qvel_off=0, control=True):
        """Advance one control step. Returns torques[29] (full robot)."""
        qpos = data.qpos[qpos_off: qpos_off + 36]
        qvel = data.qvel[qvel_off: qvel_off + 35]
        # balance base target (legs+waist) — LocoBase29.update reads full qpos/qvel
        self.loco.update(qpos, qvel)
        leg_target = self.loco.target.copy()  # (29,) absolute targets

        # arm target blend
        if self.active is not None and self.active in self.clips:
            clip = self.clips[self.active]
            T = clip["arm"].shape[0]
            fi = int(self.active_t * clip["fps"]) % T
            strike_arm = clip["arm"][fi] - HOME[ARMS]  # relative to HOME
            self.arm_target += self.blend * (strike_arm - self.arm_target)
            self.active_t += self.control_dt
        else:
            self.arm_target += self.blend * (STANCE_ARMS - self.arm_target)
        # write arms into the combined target
        leg_target[ARMS] = HOME[ARMS] + self.arm_target

        tau = self.loco.pd_torque(qpos, qvel, target_override=leg_target)
        tau = np.clip(tau, -120, 120)
        if not np.all(np.isfinite(tau)):
            tau = np.zeros(29, dtype=np.float32)
        return tau

    def reset(self):
        self.loco.reset()
        self.active = None
        self.active_t = 0.0
        self.arm_target = np.zeros(14, dtype=np.float32)
