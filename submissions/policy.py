"""FightLab v6 policy bundle — contract v2 (73D obs -> 29D joint targets).

Structure:
  PPO actor (73 -> 32D latent residual delta_z)
  + frozen prior  (73 -> z_mu)
  + frozen decoder ((s_prop 61, z) -> 29 joint targets in [-1,1])
  -> joint position targets = guard_pose + action * ACTION_SCALE

The bundle is self-contained: it loads both the PPO checkpoint and the
distilled latent checkpoint. No external deps beyond torch + numpy.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM = 73          # ang_vel(3) + jp_rel(29) + jv(29) + goal(12)
S_PROP_DIM = 61       # ang_vel(3) + jp_rel(29) + jv(29)  (first 61 of obs)
LATENT_DIM = 32
ACT_DIM = 29
ACTION_SCALE = 0.25   # must match training LatentResidualActionCfg.action_scale
PRIOR_SCALE = 0.5     # z = z_mu * prior_scale + delta_z (training convention)

# Guard pose in JOINT_NAMES_29 order (contract v2).
DEFAULT_POSE_29 = np.array([
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,
    0.0, 0.0, 0.0,
    0.20, 0.20, 0.0, 0.9, 0.0, 0.0, 0.0,
    0.20, -0.20, 0.0, 0.9, 0.0, 0.0, 0.0,
], dtype=np.float32)


def _build_mlp(input_dim, output_dim, hidden, output_activation=None):
    layers = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class _GaussianHead(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.mu_head = nn.Linear(feature_dim, latent_dim)
        self.log_sigma_head = nn.Linear(feature_dim, latent_dim)

    def forward(self, x):
        mu = self.mu_head(x)
        log_sigma = torch.clamp(self.log_sigma_head(x), min=-5.0, max=2.0)
        sigma = F.softplus(log_sigma) + 1e-4
        return mu, sigma


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _build_mlp(S_PROP_DIM + LATENT_DIM, ACT_DIM, [1024, 1024, 1024], output_activation="tanh")

    def forward(self, s_prop, z):
        return self.net(torch.cat([s_prop, z], dim=-1))


class _Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _build_mlp(S_PROP_DIM, 1024, [1024, 1024])
        self.head = _GaussianHead(1024, LATENT_DIM)

    def forward(self, s_prop):
        return self.head(self.trunk(s_prop))


class Policy:
    """Contract-conformant policy: predict(obs73) -> 29 joint position targets."""

    def __init__(self, bundle_dir: str):
        device = torch.device("cpu")

        # Frozen latent decoder + prior (distilled 29-DoF motion manifold).
        lat = torch.load(os.path.join(bundle_dir, "latent_29.pt"), map_location="cpu", weights_only=False)
        self.decoder = _Decoder()
        self.decoder.load_state_dict(lat["decoder"])
        self.decoder.eval()
        self.prior = _Prior()
        self.prior.load_state_dict(lat["prior"])
        self.prior.eval()

        # PPO actor (73 -> 32D latent residual delta).
        ckpt = torch.load(os.path.join(bundle_dir, "policy.pt"), map_location="cpu", weights_only=False)
        actor_sd = ckpt["actor_state_dict"]
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, LATENT_DIM),
        )
        self.actor.load_state_dict({k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")})
        self.actor.eval()

        # Observation normalizer from training.
        self.obs_mean = actor_sd["obs_normalizer._mean"]
        self.obs_var = actor_sd["obs_normalizer._var"]

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        # Normalize obs as in training.
        obs_n = (obs_t - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        delta_z = self.actor(obs_n)
        s_prop = obs_t[:, :S_PROP_DIM]
        z_mu, _ = self.prior(s_prop)
        z = F.normalize(PRIOR_SCALE * z_mu + delta_z, p=2, dim=-1)
        action = self.decoder(s_prop, z)  # (1, 29) in [-1,1]
        target = DEFAULT_POSE_29 + action.squeeze(0).numpy() * ACTION_SCALE
        return target.astype(np.float32)


def load(path: str) -> Policy:
    """Bundle entry point (contract): return a Policy with .predict(obs)."""
    return Policy(path)
