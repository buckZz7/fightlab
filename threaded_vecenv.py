"""Threaded VecEnv: N envs in ONE process, stepped on a thread pool.

Why: SubprocVecEnv costs one full MuJoCo scene + ONNX session per worker
(~600MB on G1) plus IPC serialization of every obs/action. On this 3.9GB
box that caps at 4 workers and sync overhead gutted throughput (~2h for
<100k steps).

Design:
- One process, N G1PunchEnv instances (each owns its MuJoCo model+data —
  physics state can't be shared)
- ONE shared ONNX session across all envs' LocoBase29 instances (inference
  is stateless given obs; ort releases the GIL during run())
- step_async fans out env.step on ThreadPoolExecutor; mj_step + onnx run
  release the GIL, so physics parallelizes across cores
- SB3 VecEnv interface, drop-in replacement

Expected: ~1.2GB total for 8 envs (vs OOM at 8 processes), no IPC cost.
"""
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Type

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


class ThreadedVecEnv(VecEnv):
    def __init__(self, env_fns: List[Type], max_workers: Optional[int] = None):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        self.num_envs = len(self.envs)
        super().__init__(
            self.num_envs,
            self.envs[0].observation_space,
            self.envs[0].action_space,
        )
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers or min(self.num_envs, mp.cpu_count()))
        self._actions = None

    def reset(self):
        return np.stack([env.reset()[0] for env in self.envs])

    def step_async(self, actions: np.ndarray):
        self._actions = actions

    def step_wait(self):
        futures = [self.pool.submit(env.step, action)
                   for env, action in zip(self.envs, self._actions)]
        results = [f.result() for f in futures]
        obs, rews, terms, truncs, infos = zip(*results)
        dones = np.logical_or(terms, truncs)
        new_obs = list(obs)
        for i, done in enumerate(dones):
            if done:
                new_obs[i] = self.envs[i].reset()[0]
                infos[i]["terminal_observation"] = obs[i]
        return (np.stack(new_obs), np.array(rews), dones, list(infos))

    def close(self):
        self.pool.shutdown(wait=True)
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(env, attr_name, None) for env in self.envs]

    def set_attr(self, attr_name, value, indices=None):
        for env in self.envs:
            setattr(env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None,
                   **method_kwargs):
        return [getattr(env, method_name)(*method_args, **method_kwargs)
                for env in self.envs]

    def env_is_wrapped(self, wrapper_class, indices=None):
        from stable_baselines3.common import vec_env
        return [vec_env.is_wrapped(env, wrapper_class) for env in self.envs]

    def seed(self, seed: Optional[int] = None):
        return [env.reset(seed=(seed + i if seed is not None else None))[1]
                for i, env in enumerate(self.envs)]


def make_threaded_vec_env(env_fn, n_envs: int, max_workers: Optional[int] = None):
    return ThreadedVecEnv([env_fn for _ in range(n_envs)], max_workers=max_workers)
