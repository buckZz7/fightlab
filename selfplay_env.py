"""Self-play boxing harness: challenger (learning) vs opponent (frozen policy).

Wraps BoxingEnv as a single-agent gymnasium.Env so SB3 PPO can train the
challenger. The opponent is a frozen SB3 policy (or None for random torques).

Stage 2: opponent = frozen punch/bag policy (or random) to seed the league.
Stage 3: opponent = frozen previous king. Beat the king -> you are the king.

Action space: 9 joint torques in [-1, 1], scaled to the env's +/-50 N m.
Observation: BoxingEnv._get_agent_obs (47-dim float vector).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from boxing_env import BoxingEnv

ACTION_SCALE = 50.0  # env clips ctrl to +/-50; policy outputs [-1, 1]
OBS_DIM = 45
ACT_DIM = 9


class SelfPlayEnv(gym.Env):
    """Single-agent view of BoxingEnv for PPO training."""

    metadata = {"render_modes": []}

    def __init__(self, opponent_model=None, opponent_agent=2,
                 max_steps=2000, randomize=True):
        super().__init__()
        self.env = BoxingEnv(randomize=randomize, max_steps=max_steps)
        self.opponent = opponent_model      # SB3 model w/ .predict, or None
        self.opponent_agent = opponent_agent
        self.me = 3 - opponent_agent        # challenger is the other agent
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)

    def _opp_action(self, obs_dict):
        key = f"agent_{self.opponent_agent}"
        if self.opponent is None:
            return np.random.uniform(-1, 1, ACT_DIM) * ACTION_SCALE
        a, _ = self.opponent.predict(obs_dict[key], deterministic=True)
        return np.clip(a, -1, 1) * ACTION_SCALE

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return obs[f"agent_{self.me}"].astype(np.float64), {}

    def step(self, action):
        obs = self.env._get_obs()
        opp_raw = self._opp_action(obs)
        actions = {
            f"agent_{self.me}": np.clip(action, -1, 1) * ACTION_SCALE,
            f"agent_{self.opponent_agent}": opp_raw,
        }
        obs, rewards, done, info = self.env.step(actions)
        my_key = f"agent_{self.me}"
        reward = rewards[my_key]
        # Win/loss shaping at episode end
        if done:
            my_hp = info[f"hp_{self.me}"]
            opp_hp = info[f"hp_{self.opponent_agent}"]
            if opp_hp <= 0 or (my_hp > opp_hp):
                reward += 25.0   # win bonus
            elif my_hp <= 0 or (my_hp < opp_hp):
                reward -= 25.0   # loss penalty
        return (obs[my_key].astype(np.float64), reward, done, False, info)


def make_selfplay_env(opponent_path=None, opponent_agent=2, **kw):
    from stable_baselines3 import PPO
    opp = PPO.load(opponent_path) if opponent_path else None
    return SelfPlayEnv(opponent_model=opp, opponent_agent=opponent_agent, **kw)
