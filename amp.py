"""AMP: Adversarial Motion Prior discriminator.

RoboStriker showed that dropping AMP reduces hit rate from 0.685 to 0.49.
AMP adds a style reward that keeps the fighter's motion looking natural
(like the mocap) instead of degenerating into robotic flailing.

Lightweight implementation: a simple MLP discriminator that distinguishes
real mocap motion from policy-generated motion. The reward = how much
the policy's motion looks like real combat.
"""
import numpy as np
import torch
import torch.nn as nn


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
            # Clamp to avoid log(0)
            d = max(0.01, min(0.99, d))
            return -np.log(1.0 - d)

    def train_step(self, real_obs, fake_obs, optimizer):
        """Train discriminator: real=1, fake=0.

        real_obs: list of mocap joint states (58-dim each)
        fake_obs: list of policy-generated joint states
        """
        real = torch.FloatTensor(np.array(real_obs))
        fake = torch.FloatTensor(np.array(fake_obs))

        # Real samples -> target 1
        pred_real = self.net(real).squeeze(-1)
        loss_real = nn.functional.binary_cross_entropy(pred_real, torch.ones_like(pred_real))

        # Fake samples -> target 0
        pred_fake = self.net(fake).squeeze(-1)
        loss_fake = nn.functional.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))

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
        # Extract current joint state
        off = env.qpos_off
        qp = env.data.qpos[off + 7:off + 36] - env.home
        qv = env.data.qvel[off + 6:off + 35]
        samples.append(np.concatenate([qp, qv]))
    return samples
