"""Motion Tracker: Stage 1 of the RoboStriker-style combat pipeline.

Trains a single-agent RL policy that learns to track retargeted G1
mocap motions (from exptech/g1-moves dataset). The policy outputs
target joint positions for PD control, and is rewarded for matching
the reference trajectory.

This gives the fighter physical combat skills (punching, dodging,
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


# ===========================================================================
# amp.py — Adversarial Motion Prior discriminator (inlined; only used here)
# ===========================================================================
# AMP: Adversarial Motion Prior discriminator.
#
# RoboStriker showed that dropping AMP reduces hit rate from 0.685 to 0.49.
# AMP adds a style reward that keeps the fighter's motion looking natural
# (like the mocap) instead of degenerating into robotic flailing.
#
# Lightweight implementation: a simple MLP discriminator that distinguishes
# real mocap motion from policy-generated motion. The reward = how much
# the policy's motion looks like real combat.
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # torch optional; AMP just disabled if missing
    _HAS_TORCH = False


if _HAS_TORCH:

    class AMPDiscriminator(nn.Module):
        """Simple MLP discriminator: real mocap vs policy motion.

        Input: joint positions + velocities (58-dim = 29 pos + 29 vel)
        Output: 1 (real) or 0 (fake) probability
        """
        def __init__(self, input_dim=58, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

        def compute_reward(self, obs):
            """AMP reward: how much this motion looks like real combat.

            reward = -log(1 - D(obs)) — high when D thinks it's real.
            """
            with torch.no_grad():
                x = torch.FloatTensor(obs).unsqueeze(0)
                d = self.net(x).item()
                d = max(0.01, min(0.99, d))
                return -np.log(1.0 - d)

        def train_step(self, real_obs, fake_obs, optimizer):
            """Train discriminator: real=1, fake=0.

            real_obs: list of mocap joint states (58-dim each)
            fake_obs: list of policy-generated joint states
            """
            real = torch.FloatTensor(np.array(real_obs))
            fake = torch.FloatTensor(np.array(fake_obs))

            pred_real = self.net(real).squeeze(-1)
            loss_real = nn.functional.binary_cross_entropy(
                pred_real, torch.ones_like(pred_real))

            pred_fake = self.net(fake).squeeze(-1)
            loss_fake = nn.functional.binary_cross_entropy(
                pred_fake, torch.zeros_like(pred_fake))

            loss = loss_real + loss_fake
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss.item()


    def collect_mocap_samples(motions, n_samples=1000):
        """Collect real mocap joint states samples for AMP training.

        Returns list of (29 pos + 29 vel) = 58-dim vectors.
        """
        samples = []
        for motion in motions:
            jp = motion["joint_pos"]
            jv = motion["joint_vel"]
            T = len(jp)
            for _ in range(n_samples // len(motions)):
                t = np.random.randint(0, T)
                samples.append(np.concatenate([jp[t], jv[t]]))
        return samples


    def collect_policy_samples(env, model, n_samples=1000):
        """Collect policy-generated joint state samples."""
        samples = []
        obs, _ = env.reset()
        for _ in range(n_samples):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
            off = env.qpos_off
            qp = env.data.qpos[off + 7:off + 36] - env.home
            qv = env.data.qvel[off + 6:off + 35]
            samples.append(np.concatenate([qp, qv]))
        return samples

else:  # torch not available — stubs so imports never fail
    class AMPDiscriminator:
        def __init__(self, *a, **k):
            raise RuntimeError("AMP requires torch, which is not installed.")
    def collect_mocap_samples(*a, **k):
        raise RuntimeError("AMP requires torch, which is not installed.")
    def collect_policy_samples(*a, **k):
        raise RuntimeError("AMP requires torch, which is not installed.")


# ===========================================================================
# Motion tracker env + training (was train_motion_tracker.py)
# ===========================================================================
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

        # Stay upright — HEAVY penalty for falling
        pelvis_z = self.data.xpos[pelvis_id][2]
        upright_reward = 2.0 * np.exp(-50.0 * max(0.0, 0.793 - pelvis_z))

        # Foot contact reward (both feet on ground = stable)
        foot_contact = 0
        for side in ("left", "right"):
            fb = self.model.body(f"r1_{side}_ankle_roll_link").id
            if self.data.xpos[fb][2] < 0.1:
                foot_contact += 1
        foot_reward = 0.1 * foot_contact

        reward = 0.25 * joint_reward + 0.15 * vel_reward + 0.2 * body_reward + 0.3 * upright_reward + 0.1 * foot_reward

        # AMP style reward: how much this motion looks like real combat
        # (optional — only if AMP discriminator is loaded)
        if hasattr(self, "amp") and self.amp is not None:
            cur_jp = self.data.qpos[self.qpos_off + 7:self.qpos_off + 36] - self.home
            cur_jv = self.data.qvel[self.qvel_off + 6:self.qvel_off + 35]
            amp_obs = np.concatenate([cur_jp, cur_jv])
            amp_reward = self.amp.compute_reward(amp_obs)
            reward += 0.1 * amp_reward

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
    ap.add_argument("--amp", action="store_true", help="enable AMP style reward")
    a = ap.parse_args()

    motions = load_motions(a.data)
    if not motions:
        print(f"No motions found in {a.data}")
        sys.exit(1)

    # Initialize AMP discriminator if enabled
    amp_disc = None
    if a.amp:
        # AMPDiscriminator is now defined in this module (inlined from amp.py)
        amp_disc = AMPDiscriminator(input_dim=58)
        real_samples = collect_mocap_samples(motions, n_samples=2000)
        print(f"[tracker] AMP enabled with {len(real_samples)} real samples")
        # Pre-train discriminator on real samples
        import torch
        opt = torch.optim.Adam(amp_disc.parameters(), lr=1e-3)
        fake_initial = [np.random.randn(58) * 0.5 for _ in range(len(real_samples))]
        for _ in range(100):
            amp_disc.train_step(real_samples, fake_initial, opt)
        print("[tracker] AMP discriminator pre-trained")

    def _init_amp():
        env = MotionTrackerEnv(motions, max_steps=a.max_steps)
        env.amp = amp_disc
        env.reset(seed=0)
        return env

    if a.amp:
        env = SubprocVecEnv([_init_amp for _ in range(a.envs)])
    else:
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
