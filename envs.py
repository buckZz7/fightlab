"""Balance-under-contact env: MuJoCo humanoid, random torso pushes.

The first milestone of the fight league: can a policy stay upright while
being shoved at random times, directions, and magnitudes? This is the
primitive that striking-and-getting-struck is built on.
"""
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.envs.mujoco import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.4)),
    "elevation": -15.0,
}


class BalanceEnv(HumanoidEnv):
    """Humanoid standing task with randomized external pushes.

    Reward: alive bonus + uprightness, minus control cost and fall.
    Pushes: every push_interval +/- jitter steps, a random horizontal
    impulse is applied to the torso for a few steps.
    """

    def __init__(
        self,
        push_interval=125,          # ~0.5s at 5x frame skip of 0.003 dt... (15 steps/s? no: frame_skip=5, dt=0.003 -> 66 steps/s)
        push_jitter=75,
        push_force_range=(100.0, 400.0),  # Newtons
        push_duration=5,            # steps the force is applied
        healthy_z_range=(0.6, 2.0),  # 0.6 = crouching is legal, actual fall is not
        reset_noise_scale=1e-2,
        crouch_penalty=0.5,          # reward penalty for torso below this height
        upright_target=1.28,         # nominal standing torso height
        **kwargs,
    ):
        super().__init__(
            terminate_when_unhealthy=True,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=False,
            **kwargs,
        )
        self._push_interval = push_interval
        self._push_jitter = push_jitter
        self._push_force_range = push_force_range
        self._push_duration = push_duration
        self._crouch_penalty = crouch_penalty
        self._upright_target = upright_target
        self._steps_to_next_push = 0
        self._push_steps_left = 0
        self._push_vec = np.zeros(3)
        self.pushes_survived = 0
        self.pushes_total = 0

    def _schedule_push(self):
        self._steps_to_next_push = self._push_interval + self.np_random.integers(
            -self._push_jitter, self._push_jitter)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._schedule_push()
        self._push_steps_left = 0
        self.pushes_survived = 0
        self.pushes_total = 0
        return obs, info

    def step(self, action):
        # push logic: count down, then apply force for push_duration steps
        self._steps_to_next_push -= 1
        if self._steps_to_next_push <= 0 and self._push_steps_left == 0:
            mag = self.np_random.uniform(*self._push_force_range)
            ang = self.np_random.uniform(0, 2 * np.pi)
            self._push_vec = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            self._push_steps_left = self._push_duration
            self.pushes_total += 1
            self._schedule_push()

        if self._push_steps_left > 0:
            torso_id = self.model.body("torso").id
            self.data.xfrc_applied[torso_id, :3] = self._push_vec
            self._push_steps_left -= 1
            if self._push_steps_left == 0:
                self._push_vec = np.zeros(3)
                self.pushes_survived += 1  # survived if we get here w/o termination... refined below
        else:
            torso_id = self.model.body("torso").id
            self.data.xfrc_applied[torso_id, :] = 0.0

        obs, reward, terminated, truncated, info = super().step(action)
        # crouch penalty: torso height below upright target costs reward
        torso_z = float(self.data.body("torso").xpos[2])
        crouch = max(0.0, self._upright_target - torso_z)
        reward -= self._crouch_penalty * crouch
        if terminated and self._push_steps_left > 0:
            self.pushes_survived -= 1  # fell mid-push: doesn't count as survived
        info["pushes_survived"] = self.pushes_survived
        info["pushes_total"] = self.pushes_total
        info["torso_z"] = torso_z
        return obs, reward, terminated, truncated, info


def make_env(render_mode=None, **kwargs):
    return BalanceEnv(render_mode=render_mode, **kwargs)
