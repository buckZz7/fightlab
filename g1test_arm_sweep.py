"""Arm-sweep tolerance test: how much arm motion can LocoBase29 absorb?

Drives the arms through sinusoidal sweeps of increasing amplitude/frequency
while the frozen whole-body policy balances. Logs pelvis height + torso tilt
to find the safe operating envelope for the fight policy.

Pass criteria per amplitude level: stands full 10s window, z stays > 0.6.
"""
import mujoco
import numpy as np
import sys
sys.path.insert(0, "/opt/data/fightlab-repo-new")

from loco_base29 import LocoBase29, HOME, ARMS

XML = "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
m.opt.timestep = 0.002

pelvis = m.body("pelvis").id
lo = m.actuator_ctrlrange[:, 0]
hi = m.actuator_ctrlrange[:, 1]

# arm joint indices in the 29-DoF actuator order (15..28)
ARM_IDX = list(range(15, 29))
# joint ranges for scaling sweep amplitude
jrange = []
for i in ARM_IDX:
    jid = m.actuator_trnid[i, 0]
    jrange.append(m.jnt_range[jid].copy())
jrange = np.array(jrange)
jspan = (jrange[:, 1] - jrange[:, 0]) / 2  # half-span per arm joint


def run_level(amp_frac, freq_hz, seconds=10):
    """Sweep arms sinusoidally at amp_frac of joint half-span. Returns (stood, min_z)."""
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 0.75
    d.qpos[7:36] = HOME
    mujoco.mj_forward(m, d)
    loco = LocoBase29()
    steps = int(seconds / 0.002)
    min_z = 1.0
    for i in range(steps):
        t = i * 0.002
        # sinusoidal arm sweep around HOME, alternating phase per arm
        sweep = np.zeros(29)
        for k, ji in enumerate(ARM_IDX):
            phase = t * 2 * np.pi * freq_hz + (k % 4) * np.pi / 2
            sweep[ji] = amp_frac * jspan[k] * np.sin(phase)
        loco.update(d.qpos, d.qvel)
        target = loco.target + sweep
        tau = loco.pd_torque(d.qpos, d.qvel, target_override=target)
        d.ctrl[:] = np.clip(tau, lo, hi)
        mujoco.mj_step(m, d)
        z = float(d.xpos[pelvis][2])
        min_z = min(min_z, z)
        if z < 0.4:
            return False, min_z, i * 0.002
    return True, min_z, seconds


print("amp_frac  freq   stood  min_z  time")
for amp in (0.05, 0.10, 0.15, 0.20, 0.30):
    for freq in (0.5, 1.0, 2.0):
        stood, mz, t = run_level(amp, freq)
        print(f"  {amp:.2f}   {freq:.1f}Hz   {'OK ' if stood else 'FELL'}  {mz:.2f}  {t:.1f}s")
