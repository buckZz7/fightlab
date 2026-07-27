"""Motion Tracker: Stage 1 of the RoboStriker-style boxing pipeline.

Trains a single-agent RL policy that learns to track retargeted G1
mocap motions (from exptech/g1-moves dataset). The policy outputs
target joint positions for PD control, and is rewarded for matching
the reference trajectory.

This gives the fighter physical boxing skills (punching, dodging,
footwork) as a foundation before adding combat rewards.

Training on the EGL pod (GPU + fast render):
  python3 train_motion_tracker.py --data /workspace/g1-moves/karate \
      --steps 500000 --out models/motion_tracker
"""
import os, sys, argparse, glob, random
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

DT = 0.002
FRAME_SKIP = 4  # control at 125Hz (physics 500Hz / 4)
CONTROL_DT = DT * FRAME_SKIP  # 0.008s per control step
N_DOFS = 29


def load_motions(data_dir):
    """Load all .npz training files from the dataset."""
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.npz"), recursive=True))
    motions = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        motions.append({
            "joint_pos": d["joint_pos"],     # (T, 29) target joint angles
            "joint_vel": d["joint_vel"],     # (T, 29) target joint velocities
            "body_pos": d["body_pos_w"],     # (T, 30, 3) link world positions
            "body_quat": d["body_quat_w"],   # (T, 30, 4) link world quats
            "fps": float(d["fps"][0]) if hasattr(d["fps"], "__len__") else float(d["fps"]),
            "name": os.path.basename(f).replace(".npz", ""),
        })
    print(f"[tracker] loaded {len(motions)} motions from {data_dir}")
    return motions


class MotionTrackerEnv(gym.Env):
    """RL env: G1 tracks a reference motion. Reward = joint + body match."""

    def __init__(self, motions, max_steps=1000, randomize=True):
        super().__init__()
        self.motions = motions
        self.max_steps = max_steps
        self.randomize = randomize

        # Load the G1 model (single robot)
        from street_arena import build_default_2bot
        self.model = build_default_2bot()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT

        # We only use r1 (robot 0)
        self.qpos_off = 0
        self.qvel_off = 0
        self.ctrl_slice = slice(0, 29)

        # Home pose (from the model's default qpos)
        self.home = self.data.qpos[self.qpos_off + 7:self.qpos_off + 36].copy()
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

        # Action: target joint position deltas (29 DoF)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(N_DOFS,), dtype=np.float32)

        # Observation: current joint state + reference motion (window)
        obs_dim = N_DOFS * 2 + N_DOFS * 2  # current pos+vel + target pos+vel
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.frame_skip = FRAME_SKIP
        self.step_count = 0
        self.motion = None
        self.motion_t = 0
        self._rng = np.random.RandomState()

    def _get_obs(self):
        """Current joint state + reference target."""
        qp = self.data.qpos[self.qpos_off + 7:self.qpos_off + 36]
        qv = self.data.qvel[self.qvel_off + 6:self.qvel_off + 35]
        # Reference: current + next frame
        t = min(self.motion_t, len(self.motion["joint_pos"]) - 1)
        ref_pos = self.motion["joint_pos"][t]
        ref_vel = self.motion["joint_vel"][t]
        return np.concatenate([qp - self.home, qv, ref_pos - self.home, ref_vel]).astype(np.float32)

    def _randomize(self):
        """Domain randomization for sim2real robustness."""
        self.model.body_mass[:] = self._base_mass * self._rng.uniform(0.9, 1.1, self.model.nbody)
        self.model.geom_friction[:, 0] = self._base_friction[:, 0] * self._rng.uniform(0.85, 1.15, self.model.ngeom)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        # Pick a random motion
        self.motion = random.choice(self.motions)
        # Pick a random start time within the motion (reference state init)
        T = len(self.motion["joint_pos"])
        self.motion_t = random.randint(0, max(0, T - 100))

        # Set robot to the reference pose at start time
        self.data.qpos[self.qpos_off:self.qpos_off + 3] = [0, 0, 0.793]
        self.data.qpos[self.qpos_off + 3:self.qpos_off + 7] = [1, 0, 0, 0]
        self.data.qpos[self.qpos_off + 7:self.qpos_off + 36] = self.motion["joint_pos"][self.motion_t]

        # Domain randomization
        if self.randomize:
            self._randomize()

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # PD control: target = home + action * scale
        target = self.home + action * 0.5  # scale action to joint range
        kp, kd = 100.0, 5.0

        for _ in range(self.frame_skip):
            tau = kp * (target - self.data.qpos[self.qpos_off + 7:self.qpos_off + 36]) \
                - kd * self.data.qvel[self.qvel_off + 6:self.qvel_off + 35]
            tau = np.clip(tau, -88, 88)
            self.data.ctrl[self.ctrl_slice] = tau
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1
        self.motion_t = min(self.motion_t + 1, len(self.motion["joint_pos"]) - 1)

        # Reward: joint position tracking + joint velocity tracking + body + upright
        t = self.motion_t
        ref_pos = self.motion["joint_pos"][t]
        ref_vel = self.motion["joint_vel"][t]
        cur_pos = self.data.qpos[self.qpos_off + 7:self.qpos_off + 36]
        cur_vel = self.data.qvel[self.qvel_off + 6:self.qvel_off + 35]

        joint_err = np.mean((cur_pos - ref_pos) ** 2)
        joint_reward = np.exp(-10.0 * joint_err)

        vel_err = np.mean((cur_vel - ref_vel) ** 2)
        vel_reward = np.exp(-2.0 * vel_err)

        # Body position tracking (pelvis)
        ref_body = self.motion["body_pos"][t]
        pelvis_id = self.model.body("r1_pelvis").id
        cur_pelvis = self.data.xpos[pelvis_id]
        ref_pelvis = ref_body[0]
        body_err = np.mean((cur_pelvis - ref_pelvis) ** 2)
        body_reward = np.exp(-5.0 * body_err)

        # Stay upright
        pelvis_z = self.data.xpos[pelvis_id][2]
        upright_reward = 1.0 if pelvis_z > 0.5 else -1.0

        reward = 0.3 * joint_reward + 0.2 * vel_reward + 0.3 * body_reward + 0.2 * upright_reward

        # Penalize falling
        terminated = pelvis_z < 0.3
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, {}


def make_env(motions, max_steps, seed):
    def _init():
        env = MotionTrackerEnv(motions, max_steps=max_steps)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/g1-moves/karate")
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--out", default="models/motion_tracker")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=500)
    a = ap.parse_args()

    motions = load_motions(a.data)
    if not motions:
        print(f"No motions found in {a.data}")
        sys.exit(1)

    env = SubprocVecEnv([make_env(motions, a.max_steps, i) for i in range(a.envs)])

    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                verbose=1)

    print(f"[tracker] training {a.steps} steps with {a.envs} envs...")
    model.learn(total_timesteps=a.steps)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    model.save(a.out)
    print(f"[tracker] saved {a.out}")


if __name__ == "__main__":
    main()
