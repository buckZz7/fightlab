"""Frozen whole-body G1 balance base with 180° Z rotation wrapper.

LocoBase29Rot wraps LocoBase29 for a robot rotated 180° around Z so the
balance policy sees a forward-facing robot and produces correct torques.

For 180° Z rotation:
  - gravity in body frame: UNCHANGED (body z still points up)
  - angular velocity: flip x and y (body x/y are mirrored)
  - cmd: negate vx, vy (world-frame commands rotated 180°)
  - joint positions/velocities: UNCHANGED (body-local)
  - action output: swap L/R limbs, negate lateral joints

The L/R swap is needed because the policy says "step forward with left
leg" but the rotated robot's left leg is now on the right side.
"""
import numpy as np
from loco_base29 import LocoBase29, HOME, SCALE, KP, KD, DECIMATION, GAIT_PERIOD

# Joint swap: L leg(0-5) <-> R leg(6-11), L arm(15-21) <-> R arm(22-28)
# Sign flip on lateral joints (roll, yaw) that mirror under 180° Z
def _build_swap():
    swap = list(range(29))
    sign = [1.0] * 29
    for i in range(6):
        swap[i], swap[i+6] = swap[i+6], swap[i]
    # Negate lateral leg joints: hip_roll(1,7), hip_yaw(2,8), ankle_roll(5,11)
    for i in [1, 2, 5, 7, 8, 11]:
        sign[i] = -1.0
    # Waist: negate yaw(12) and roll(13)
    sign[12] = -1.0
    sign[13] = -1.0
    # Swap L arm(15-21) <-> R arm(22-28)
    for i in range(7):
        swap[15+i], swap[22+i] = swap[22+i], swap[15+i]
    # Negate shoulder_roll(16,23), wrist_roll(20,27)
    for i in [16, 20, 23, 27]:
        sign[i] = -1.0
    return np.array(swap), np.array(sign)

SWAP_IDX, SWAP_SIGN = _build_swap()


class LocoBase29Rot(LocoBase29):
    """LocoBase29 for a robot rotated 180° around Z (faces opposite direction)."""

    def update(self, qpos, qvel):
        self._counter += 1
        if self._counter % DECIMATION != 0:
            return self.target

        phase = (self._counter * 0.002 % GAIT_PERIOD) / GAIT_PERIOD

        # For 180° Z: gravity unchanged, flip omega x/y, negate cmd vx/vy
        omega = qvel[3:6]
        omega_flipped = np.array([-omega[0], -omega[1], omega[2]])
        cmd_flipped = np.array([-self.cmd[0], -self.cmd[1], self.cmd[2]])

        obs = np.concatenate([
            omega_flipped,
            self._grav_ori(qpos[3:7]),  # unchanged for 180 Z
            cmd_flipped,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            qpos[7:36] - HOME,
            qvel[6:35],
            self.action,
        ]).astype(np.float32)

        self.action = self.sess.run(None, {self.inp: obs[None]})[0].squeeze()

        # Swap L/R limbs and negate lateral joints in the output
        delta = self.action * SCALE  # delta from HOME
        delta_rot = np.zeros(29)
        for i in range(29):
            delta_rot[i] = delta[SWAP_IDX[i]] * SWAP_SIGN[i]
        self.target = delta_rot + HOME

        return self.target

    def pd_torque(self, qpos, qvel, target_override=None):
        t = self.target if target_override is None else target_override
        return KP * (t - qpos[7:36]) - KD * qvel[6:35]
