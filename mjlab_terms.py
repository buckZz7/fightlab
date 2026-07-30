"""MDP terms for the FightLab two-G1 combat warmup task.

Ports the Isaac Lab DirectRLEnv warmup env (warmup_env.py) to mjlab
manager-based terms:

- 32D residual-latent action term with frozen decoder + prior
  (loaded from /workspace/latent_final.pt).
- 61D observation: ang_vel(3) + joint_pos_rel(23) + joint_vel(23) + goal(12).
- Combat reward: facing + approach velocity + fist distance + reach + hit
  - fall penalty, plus optional AMP style reward.
- Two-robot reset event: robots spawn 0.6m apart facing each other.
- Pelvis-height terminations for both robots.

The Isaac environment multiplied the reward by the *physics* dt per physics
step; mjlab's RewardManager multiplies by the *policy* dt once per step, so
the combat reward term returns ``reward / decimation`` to match the Isaac
magnitude exactly (policy_dt * (r / decimation) == physics_dt * r).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

# --------------------------------------------------------------------------- #
# Constants (must match Isaac warmup_env.py)
# --------------------------------------------------------------------------- #

JOINT_NAMES_23 = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
]

# 29-DoF standard order (mjlab G1 XML order: legs, waist, left arm + wrist,
# right arm + wrist). This is the distilled latent's joint order.
JOINT_NAMES_29 = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DEFAULT_JOINT_POS_23 = torch.tensor(
  [
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,  # left leg
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,  # right leg
    0.0, 0.0, 0.0,  # waist
    0.20, 0.20, 0.0, 0.9,  # left arm (guard)
    0.20, -0.20, 0.0, 0.9,  # right arm (guard)
  ],
  dtype=torch.float32,
)

# Guard pose in JOINT_NAMES_29 order (wrists at 0).
DEFAULT_JOINT_POS_29 = torch.tensor(
  [
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,  # left leg
    -0.20, 0.0, 0.0, 0.50, -0.30, 0.0,  # right leg
    0.0, 0.0, 0.0,  # waist
    0.20, 0.20, 0.0, 0.9, 0.0, 0.0, 0.0,  # left arm (guard) + wrist
    0.20, -0.20, 0.0, 0.9, 0.0, 0.0, 0.0,  # right arm (guard) + wrist
  ],
  dtype=torch.float32,
)

WRIST_JOINT_NAMES = [
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_roll_joint",
]

# mjlab reward is scaled by policy dt; Isaac scaled by physics dt (=dt/dec).
DECIMATION = 4


# --------------------------------------------------------------------------- #
# Frozen decoder / prior (identical architecture to Isaac warmup_env.py)
# --------------------------------------------------------------------------- #


def _build_mlp(input_dim, output_dim, hidden, activation="elu", output_activation=None):
  act_cls = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}.get(activation, nn.ELU)
  layers = []
  prev = input_dim
  for h in hidden:
    layers.append(nn.Linear(prev, h))
    layers.append(act_cls())
    prev = h
  layers.append(nn.Linear(prev, output_dim))
  if output_activation == "tanh":
    layers.append(nn.Tanh())
  elif output_activation == "elu":
    layers.append(nn.ELU())
  return nn.Sequential(*layers)


class _GaussianHead(nn.Module):
  def __init__(self, feature_dim: int, latent_dim: int):
    super().__init__()
    self.mu_head = nn.Linear(feature_dim, latent_dim)
    self.log_sigma_head = nn.Linear(feature_dim, latent_dim)

  def forward(self, x):
    mu = self.mu_head(x)
    log_sigma = torch.clamp(self.log_sigma_head(x), min=-5.0, max=2.0)
    sigma = F.softplus(log_sigma) + 1e-4
    return mu, sigma


class FrozenDecoder(nn.Module):
  """Decoder D_psi: (s_prop[61], z[32]) -> action[29]."""

  def __init__(self):
    super().__init__()
    self.net = _build_mlp(93, 29, [1024, 1024, 1024], activation="elu", output_activation="tanh")

  def forward(self, s_prop, z):
    return self.net(torch.cat([s_prop, z], dim=-1))


class FrozenPrior(nn.Module):
  """Prior P_xi: s_prop[61] -> (mu[32], sigma[32])."""

  def __init__(self):
    super().__init__()
    self.trunk = _build_mlp(61, 1024, [1024, 1024], activation="elu")
    self.head = _GaussianHead(1024, 32)

  def forward(self, s_prop):
    return self.head(self.trunk(s_prop))

  def sample(self, s_prop, temperature=1.0):
    mu, sigma = self.forward(s_prop)
    eps = torch.randn_like(mu)
    z = mu + sigma * eps * temperature
    return F.normalize(z, p=2, dim=-1), mu, sigma


def _load_frozen_modules(ckpt_path: str, device: torch.device):
  ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  decoder = FrozenDecoder()
  decoder.load_state_dict(ckpt["decoder"])
  decoder.to(device).eval()
  for p in decoder.parameters():
    p.requires_grad_(False)
  prior = FrozenPrior()
  prior.load_state_dict(ckpt["prior"])
  prior.to(device).eval()
  for p in prior.parameters():
    p.requires_grad_(False)
  return decoder, prior


# --------------------------------------------------------------------------- #
# Quaternion helpers (copied from Isaac warmup_env.py)
# --------------------------------------------------------------------------- #


def _quat_to_rot_mat(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  n = torch.sqrt(w * w + x * x + y * y + z * z).clamp(min=1e-8)
  w, x, y, z = w / n, x / n, y / n, z / n
  c0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
  c1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
  c2 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
  return torch.stack([c0, c1, c2], dim=-1)


def _transform_to_ego(rot_mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  return torch.bmm(rot_mat.transpose(1, 2), vec.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Shared indexing helper (built once per env, cached on the env object)
# --------------------------------------------------------------------------- #


def _find_body(body_names, candidates):
  for c in candidates:
    if c in body_names:
      return body_names.index(c)
  for i, name in enumerate(body_names):
    for c in candidates:
      if c in name:
        return i
  return 0


class _CombatIdx:
  """Index book-keeping for the two-G1 combat setup, cached on env."""

  def __init__(self, env, robot_name: str, sandbag_name: str):
    device = env.device
    robot: Entity = env.scene[robot_name]
    sandbag: Entity = env.scene[sandbag_name]

    robot_joint_names = list(robot.joint_names)
    self.controlled_joint_indices = torch.tensor(
      [robot_joint_names.index(n) for n in JOINT_NAMES_29], device=device, dtype=torch.long
    )
    self.wrist_joint_indices = torch.tensor(
      [robot_joint_names.index(n) for n in WRIST_JOINT_NAMES if n in robot_joint_names],
      device=device, dtype=torch.long,
    )
    # Permutation robot controlled order -> standard order (and inverse).
    self.robot_to_std_perm = torch.tensor(
      [JOINT_NAMES_29.index(n) for n in JOINT_NAMES_29], device=device, dtype=torch.long
    )
    self.std_to_robot_perm = torch.tensor(
      [JOINT_NAMES_29.index(n) for n in JOINT_NAMES_29], device=device, dtype=torch.long
    )

    sandbag_joint_names = list(sandbag.joint_names)
    self.sandbag_controlled_indices = torch.tensor(
      [sandbag_joint_names.index(n) for n in JOINT_NAMES_29], device=device, dtype=torch.long
    )
    self.sandbag_wrist_indices = torch.tensor(
      [sandbag_joint_names.index(n) for n in WRIST_JOINT_NAMES if n in sandbag_joint_names],
      device=device, dtype=torch.long,
    )

    self.default_pose_29 = DEFAULT_JOINT_POS_29.to(device)

    # AMP discriminator judges striking style on the 23 major (non-wrist)
    # joints. Wrists are excluded (AMP was trained on 23-DoF clips).
    major = [n for n in JOINT_NAMES_29 if "wrist" not in n]
    self.amp_major_std_indices = torch.tensor(
      [JOINT_NAMES_29.index(n) for n in major], device=device, dtype=torch.long
    )

    a_body_names = list(robot.body_names)
    self.a_torso_idx = _find_body(a_body_names, ["torso_link", "chest_link", "pelvis"])
    self.a_left_wrist_idx = _find_body(
      a_body_names, ["left_wrist_pitch_link", "left_wrist_roll_link", "left_hand_link"]
    )
    self.a_right_wrist_idx = _find_body(
      a_body_names, ["right_wrist_pitch_link", "right_wrist_roll_link", "right_hand_link"]
    )
    b_body_names = list(sandbag.body_names)
    self.b_torso_idx = _find_body(b_body_names, ["torso_link", "chest_link", "pelvis"])
    self.b_pelvis_idx = b_body_names.index("pelvis") if "pelvis" in b_body_names else 0
    self.b_left_wrist_idx = _find_body(
      b_body_names, ["left_wrist_pitch_link", "left_wrist_roll_link", "left_hand_link"]
    )
    self.b_right_wrist_idx = _find_body(
      b_body_names, ["right_wrist_pitch_link", "right_wrist_roll_link", "right_hand_link"]
    )

    self.robot_name = robot_name
    self.sandbag_name = sandbag_name


def _get_idx(env, robot_name: str = "robot", sandbag_name: str = "sandbag") -> _CombatIdx:
  idx = getattr(env, "_combat_idx", None)
  if idx is None:
    idx = _CombatIdx(env, robot_name, sandbag_name)
    env._combat_idx = idx
  return idx


def _get_proprio(env, idx: _CombatIdx) -> torch.Tensor:
  """s_prop: ang_vel_body(3) + joint_pos_rel(23) + joint_vel(23) = 49."""
  robot: Entity = env.scene[idx.robot_name]
  quat = robot.data.root_link_quat_w
  rot = _quat_to_rot_mat(quat)
  ang_vel_b = _transform_to_ego(rot, robot.data.root_link_ang_vel_w)
  jp = robot.data.joint_pos[:, idx.controlled_joint_indices][:, idx.robot_to_std_perm]
  jv = robot.data.joint_vel[:, idx.controlled_joint_indices][:, idx.robot_to_std_perm]
  jp_rel = jp - idx.default_pose_29.unsqueeze(0)
  return torch.cat([ang_vel_b, jp_rel, jv], dim=-1)


# --------------------------------------------------------------------------- #
# Action term: 32D residual-latent -> 23 PD targets (robot) + standing (sandbag)
# --------------------------------------------------------------------------- #


@dataclass(kw_only=True)
class LatentResidualActionCfg(ActionTermCfg):
  """32D delta_z residual action decoded through frozen decoder+prior."""

  latent_ckpt_path: str = "/workspace/latent_29/latent_29.pt"
  prior_scale: float = 0.5
  action_scale: float = 0.25
  sandbag_name: str = "sandbag"
  # Self-play: path to a frozen opponent policy checkpoint (mjlab raw format with
  # actor_state_dict). When set, robot B is driven by that policy from B's
  # perspective instead of holding the standing pose.
  opponent_ckpt_path: str = ""

  def build(self, env) -> "LatentResidualAction":
    return LatentResidualAction(self, env)


class LatentResidualAction(ActionTerm):
  cfg: LatentResidualActionCfg

  def __init__(self, cfg: LatentResidualActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    self._idx = _get_idx(env, cfg.entity_name, cfg.sandbag_name)
    self._sandbag: Entity = env.scene[cfg.sandbag_name]
    self._robot: Entity = env.scene[cfg.entity_name]
    self._action_dim = 32
    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._decoder, self._prior = _load_frozen_modules(cfg.latent_ckpt_path, self.device)
    print(f"[Combat] Loaded frozen decoder+prior from {cfg.latent_ckpt_path}")

    # Self-play opponent (frozen policy driving robot B from its perspective).
    self._opp_actor = None
    self._opp_norm = None
    if cfg.opponent_ckpt_path:
      ck = torch.load(cfg.opponent_ckpt_path, map_location="cpu", weights_only=False)
      a = ck["actor_state_dict"]
      actor = nn.Sequential(
        nn.Linear(61, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 32),
      )
      actor.load_state_dict({k[len("mlp."):]: v for k, v in a.items() if k.startswith("mlp.")})
      actor.to(self.device).eval()
      for p_ in actor.parameters():
        p_.requires_grad_(False)
      self._opp_actor = actor
      self._opp_norm = {
        "mean": a["obs_normalizer._mean"].to(self.device),
        "var": a["obs_normalizer._var"].to(self.device),
      }
      print(f"[Combat] Self-play opponent loaded from {cfg.opponent_ckpt_path}")

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

  def apply_actions(self) -> None:
    idx = self._idx
    s_prop = _get_proprio(self._env, idx)
    with torch.no_grad():
      z_p, _, _ = self._prior.sample(s_prop)
      z = self._raw_actions + self.cfg.prior_scale * z_p
      z = F.normalize(z, p=2, dim=-1)
      action_29 = self._decoder(s_prop, z)  # (N, 29) in [-1, 1], std order

    action_robot = action_29[:, idx.std_to_robot_perm]
    target_29 = idx.default_pose_29.unsqueeze(0) + action_robot * self.cfg.action_scale

    # Agent A: full-DoF PD targets, wrists held at 0.
    target_full = self._entity.data.default_joint_pos.clone()
    target_full[:, idx.controlled_joint_indices] = target_29
    if idx.wrist_joint_indices is not None:
      target_full[:, idx.wrist_joint_indices] = 0.0
    self._entity.set_joint_position_target(target_full)

    # Agent B: self-play opponent policy if loaded, else hold standing pose.
    if self._opp_actor is not None:
      b_target = self._opponent_targets()
      sandbag_target = self._sandbag.data.default_joint_pos.clone()
      sandbag_target[:, idx.sandbag_controlled_indices] = b_target
      if idx.sandbag_wrist_indices is not None:
        sandbag_target[:, idx.sandbag_wrist_indices] = 0.0
      self._sandbag.set_joint_position_target(sandbag_target)
    else:
      sandbag_target = self._sandbag.data.default_joint_pos.clone()
      sandbag_target[:, idx.sandbag_controlled_indices] = idx.default_pose_29.unsqueeze(0)
      if idx.sandbag_wrist_indices is not None:
        sandbag_target[:, idx.sandbag_wrist_indices] = 0.0
      self._sandbag.set_joint_position_target(sandbag_target)

  def _opponent_targets(self) -> torch.Tensor:
    """Compute robot B's 23-DoF PD targets by running the frozen opponent policy
    from B's perspective. Obs mirrors A's: s_prop(B) + goal(B->A roles swapped)."""
    idx = self._idx
    robot, sandbag = self._robot, self._sandbag

    # B proprio (49D) in B's body frame.
    b_quat = sandbag.data.root_link_quat_w
    rot_b = _quat_to_rot_mat(b_quat)
    ang_vel_b = _transform_to_ego(rot_b, sandbag.data.root_link_ang_vel_w)
    jp = sandbag.data.joint_pos[:, idx.sandbag_controlled_indices]
    jv = sandbag.data.joint_vel[:, idx.sandbag_controlled_indices]
    jp_rel = jp - idx.default_pose_29.unsqueeze(0)
    s_prop_b = torch.cat([ang_vel_b, jp_rel, jv], dim=-1)

    # B goal obs (12D): offensive = A torso - B fists (B frame); defensive = A pelvis - B torso.
    a_torso = robot.data.body_link_pos_w[:, idx.a_torso_idx]
    a_pelvis = robot.data.root_link_pos_w
    b_torso = sandbag.data.body_link_pos_w[:, idx.b_torso_idx]
    b_lf = sandbag.data.body_link_pos_w[:, idx.b_left_wrist_idx]
    b_rf = sandbag.data.body_link_pos_w[:, idx.b_right_wrist_idx]
    off_l = _transform_to_ego(rot_b, a_torso - b_lf)
    off_r = _transform_to_ego(rot_b, a_torso - b_rf)
    def_l = _transform_to_ego(rot_b, a_pelvis - b_torso)
    def_r = _transform_to_ego(rot_b, a_pelvis - b_torso)
    goal_b = torch.cat([off_l, off_r, def_l, def_r], dim=-1)

    obs_b = torch.cat([s_prop_b, goal_b], dim=-1)  # (N, 61)
    obs_b = torch.clamp(
      (obs_b - self._opp_norm["mean"]) / torch.sqrt(self._opp_norm["var"] + 1e-8), -5.0, 5.0
    )

    with torch.no_grad():
      delta_z_b = self._opp_actor(obs_b)
      z_p_b, _, _ = self._prior.sample(s_prop_b)
      z_b = delta_z_b + self.cfg.prior_scale * z_p_b
      z_b = F.normalize(z_b, p=2, dim=-1)
      action_29_b = self._decoder(s_prop_b, z_b)
    return idx.default_pose_29.unsqueeze(0) + action_29_b * self.cfg.action_scale

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0


# --------------------------------------------------------------------------- #
# Observation term
# --------------------------------------------------------------------------- #


def combat_obs(env) -> torch.Tensor:
  """61D observation: s_prop(49) + goal(12)."""
  idx = _get_idx(env)
  robot: Entity = env.scene[idx.robot_name]
  sandbag: Entity = env.scene[idx.sandbag_name]

  s_prop = _get_proprio(env, idx)

  a_root_quat = robot.data.root_link_quat_w
  a_torso_pos = robot.data.body_link_pos_w[:, idx.a_torso_idx]
  a_left_wrist = robot.data.body_link_pos_w[:, idx.a_left_wrist_idx]
  a_right_wrist = robot.data.body_link_pos_w[:, idx.a_right_wrist_idx]
  b_torso_pos = sandbag.data.body_link_pos_w[:, idx.b_torso_idx]
  b_pelvis_pos = sandbag.data.body_link_pos_w[:, idx.b_pelvis_idx]

  rot_mat = _quat_to_rot_mat(a_root_quat)
  off_left_ego = _transform_to_ego(rot_mat, b_torso_pos - a_left_wrist)
  off_right_ego = _transform_to_ego(rot_mat, b_torso_pos - a_right_wrist)
  def_left_ego = _transform_to_ego(rot_mat, b_pelvis_pos - a_torso_pos)
  def_right_ego = _transform_to_ego(rot_mat, b_pelvis_pos - a_torso_pos)

  goal_obs = torch.cat([off_left_ego, off_right_ego, def_left_ego, def_right_ego], dim=-1)
  return torch.cat([s_prop, goal_obs], dim=-1)


# --------------------------------------------------------------------------- #
# Reward term (class-based, holds HP/AMP state and metrics)
# --------------------------------------------------------------------------- #


class CombatReward:
  """FightLab combat warmup reward (port of Isaac _get_rewards).

  Returns reward / DECIMATION so that after mjlab's policy-dt scaling the
  effective per-step magnitude matches the Isaac env (which scaled by the
  physics dt per substep).
  """

  def __init__(self, cfg, env):
    self._env = env
    p = cfg.params
    self.sigma_face = p.get("sigma_face", 0.3)
    self.sigma_vel = p.get("sigma_vel", 0.5)
    self.sigma_dist = p.get("sigma_dist", 0.5)
    self.w_face = p.get("w_face", 1.0)
    self.w_vel = p.get("w_vel", 2.0)
    self.w_dist = p.get("w_dist", 2.0)
    self.w_hit = p.get("w_hit", 10.0)
    self.w_fall_pen = p.get("w_fall_pen", 50.0)
    self.hit_vel_threshold = p.get("hit_vel_threshold", 0.5)
    self.contact_thresh = p.get("contact_thresh", 0.35)
    self.fall_threshold_z = p.get("fall_threshold_z", 0.5)
    # v2 anti-fall-farming: hits only count when upright and punching
    # TOWARD the opponent (projection), not falling onto them.
    self.upright_ref_z = p.get("upright_ref_z", 0.70)
    # v3: knockdown-reset-continue
    self.knockdown_z = p.get("knockdown_z", 0.4)
    self.sandbag_offset = p.get("sandbag_offset", 0.6)
    self.max_hp = p.get("max_hp", 100.0)
    self.amp_weight = p.get("amp_weight", 0.2)
    self.amp_ckpt = p.get("amp_ckpt", "/workspace/amp_pretrained_punching.pt")
    # v4 anti-collapse shaping
    self.w_knockdown_pen = p.get("w_knockdown_pen", 50.0)   # (v5: superseded by ground pen)
    self.punch_speed_ref = p.get("punch_speed_ref", 2.0)    # m/s for full hit credit
    self.reach_weight = p.get("reach_weight", 0.1)          # was hardcoded 0.3; taper
    self.sandbag_kd_credit = p.get("sandbag_kd_credit", 15.0)  # bonus for DOWNING opp while upright
    # v5: no stand-ups — ground-time penalty + gated recovery bonus
    self.w_ground_pen = p.get("w_ground_pen", 0.5)          # per step while down (~-15/s)
    self.recovery_credit = p.get("recovery_credit", 5.0)    # once per real knockdown (>=30 down steps)
    # v5.1 HumanUP-style get-up reward weights (mocap-free floor recovery)
    self.w_getup_height = p.get("w_getup_height", 1.0)      # pelvis height toward standing
    self.w_getup_progress = p.get("w_getup_progress", 5.0)  # reward height delta
    self.w_getup_upright = p.get("w_getup_upright", 1.0)    # torso upright orientation
    self.w_getup_feet = p.get("w_getup_feet", 0.5)          # feet below pelvis
    # v5.2 defensive shaping: PENALTY for taking a high-velocity hit (RoboStriker design)
    self.w_defense_pen = p.get("w_defense_pen", 8.0)         # per hit taken
    self.defense_window_steps = p.get("defense_window_steps", 10)  # (unused, kept for compat)

    n = env.num_envs
    device = env.device
    _idx0 = _get_idx(env)
    _robot0: Entity = env.scene[_idx0.robot_name]
    _bn = list(_robot0.body_names)
    self._l_foot_id = _bn.index("left_ankle_roll_link") if "left_ankle_roll_link" in _bn else _find_body(_bn, ["left_ankle_roll", "left_foot"])
    self._r_foot_id = _bn.index("right_ankle_roll_link") if "right_ankle_roll_link" in _bn else _find_body(_bn, ["right_ankle_roll", "right_foot"])
    self._hp_b = torch.full((n,), self.max_hp, device=device)
    self._amp_prev_obs = torch.zeros(n, 49, device=device)
    self._amp_prev_obs_valid = torch.zeros(n, dtype=torch.bool, device=device)
    self._amp_disc = None
    self._knockdowns = torch.zeros(n, device=device)
    self._init_amp()

  def _init_amp(self):
    try:
      import sys

      if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")
      from amp_discriminator import AMPDiscriminator

      self._amp_disc = AMPDiscriminator(
        obs_dim=49, hidden_layers=[512, 512], lr=1e-4, gp_weight=5.0,
        device=str(self._env.device),
      )
      self._amp_disc.load(self.amp_ckpt)
      print(f"[Combat AMP] Loaded pretrained discriminator from {self.amp_ckpt}")
    except Exception as e:
      print(f"[Combat AMP] Discriminator unavailable ({e}); style reward disabled")
      self._amp_disc = None

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._hp_b[env_ids] = self.max_hp
    self._knockdowns[env_ids] = 0.0
    self._amp_prev_obs[env_ids] = 0.0
    self._amp_prev_obs_valid[env_ids] = False
    if hasattr(self, "_a_was_down"):
      self._a_was_down[env_ids] = False
      self._b_was_down[env_ids] = False
      self._a_down_steps[env_ids] = 0.0
    if hasattr(self, "_za_prev"):
      self._za_prev[env_ids] = 0.75

  def __call__(self, env, **_params) -> torch.Tensor:
    idx = _get_idx(env)
    robot: Entity = env.scene[idx.robot_name]
    sandbag: Entity = env.scene[idx.sandbag_name]

    a_root_pos = robot.data.root_link_pos_w
    a_root_quat = robot.data.root_link_quat_w
    a_root_vel = robot.data.root_link_lin_vel_w
    a_left_wrist = robot.data.body_link_pos_w[:, idx.a_left_wrist_idx]
    a_right_wrist = robot.data.body_link_pos_w[:, idx.a_right_wrist_idx]
    a_left_wrist_vel = robot.data.body_link_lin_vel_w[:, idx.a_left_wrist_idx]
    a_right_wrist_vel = robot.data.body_link_lin_vel_w[:, idx.a_right_wrist_idx]

    b_root_pos = sandbag.data.root_link_pos_w
    b_torso_pos = sandbag.data.body_link_pos_w[:, idx.b_torso_idx]
    b_torso_vel = sandbag.data.body_link_lin_vel_w[:, idx.b_torso_idx]

    # Facing alignment.
    rel = b_root_pos - a_root_pos
    dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-6)
    face_dir = rel / dist.unsqueeze(-1)
    rot_mat = _quat_to_rot_mat(a_root_quat)
    forward = rot_mat[:, :, 0]
    d_x = (forward * face_dir).sum(dim=-1)
    r_face = torch.exp(-torch.clamp(1.0 - d_x, min=0.0) / self.sigma_face)

    # Approach velocity.
    approach_vel = (a_root_vel[:, :2] * face_dir[:, :2]).sum(dim=-1)
    v_tar = 1.0
    e_v = torch.clamp(v_tar - approach_vel, min=0.0)
    r_vel = torch.where(
      approach_vel > 0, torch.exp(-e_v**2 / self.sigma_vel), torch.zeros_like(approach_vel)
    )

    # Fist distance + reach.
    fist_dist_left = torch.linalg.norm(a_left_wrist - b_torso_pos, dim=-1)
    fist_dist_right = torch.linalg.norm(a_right_wrist - b_torso_pos, dim=-1)
    wrist_to_opp_left = b_torso_pos - a_left_wrist
    wrist_to_opp_right = b_torso_pos - a_right_wrist
    wtol_dir = wrist_to_opp_left / torch.linalg.norm(wrist_to_opp_left, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    wtor_dir = wrist_to_opp_right / torch.linalg.norm(wrist_to_opp_right, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    rel_vel_left = a_left_wrist_vel - b_torso_vel
    rel_vel_right = a_right_wrist_vel - b_torso_vel
    punch_speed_left = (rel_vel_left * wtol_dir).sum(dim=-1)
    punch_speed_right = (rel_vel_right * wtor_dir).sum(dim=-1)
    r_dist = (torch.exp(-fist_dist_left / self.sigma_dist) + torch.exp(-fist_dist_right / self.sigma_dist)) / 2.0
    r_reach = self.reach_weight * (torch.clamp(punch_speed_left, min=0.0) + torch.clamp(punch_speed_right, min=0.0))

    # Hit detection v2: proximity + punch velocity projected TOWARD opponent
    # (falling sideways onto the sandbag no longer counts), gated by uprightness.
    contact_left = fist_dist_left < self.contact_thresh
    contact_right = fist_dist_right < self.contact_thresh
    a_pelvis_z = a_root_pos[:, 2] - env.scene.env_origins[:, 2]

    # Per-fist punch speed (projection onto attack direction, computed above).
    hit_left = contact_left & (punch_speed_left > self.hit_vel_threshold)
    hit_right = contact_right & (punch_speed_right > self.hit_vel_threshold)
    hit_raw = hit_left | hit_right

    # Upright gate: hits scale with pelvis height (0 below 0.4, full at upright_ref).
    upright_scale = torch.clamp(
      (a_pelvis_z - 0.4) / (self.upright_ref_z - 0.4), min=0.0, max=1.0
    )
    hit = hit_raw & (upright_scale > 0.5)

    rel_vel = torch.maximum(rel_vel_left.norm(dim=-1), rel_vel_right.norm(dim=-1))
    best_punch = torch.maximum(
      torch.clamp(punch_speed_left, min=0.0), torch.clamp(punch_speed_right, min=0.0)
    )

    # Defensive shaping: PENALTY for being hit by a high-velocity strike from B
    # (RoboStriker Table 5 design — r_def is a cost, not a farmable reward).
    # B's strike counts if B's fist moves toward A's torso fast AND contacts A.
    b_lwrist = sandbag.data.body_link_pos_w[:, idx.b_left_wrist_idx]
    b_rwrist = sandbag.data.body_link_pos_w[:, idx.b_right_wrist_idx]
    b_lwrist_vel = sandbag.data.body_link_lin_vel_w[:, idx.b_left_wrist_idx]
    b_rwrist_vel = sandbag.data.body_link_lin_vel_w[:, idx.b_right_wrist_idx]
    a_torso_pos = robot.data.body_link_pos_w[:, idx.a_torso_idx]
    bw_tol_dir = a_torso_pos - b_lwrist
    bw_tol_dir = bw_tol_dir / torch.linalg.norm(bw_tol_dir, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    bw_tor_dir = a_torso_pos - b_rwrist
    bw_tor_dir = bw_tor_dir / torch.linalg.norm(bw_tor_dir, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    b_punch_left = (b_lwrist_vel * bw_tol_dir).sum(dim=-1)
    b_punch_right = (b_rwrist_vel * bw_tor_dir).sum(dim=-1)
    b_contact_left = torch.linalg.norm(b_lwrist - a_torso_pos, dim=-1) < self.contact_thresh
    b_contact_right = torch.linalg.norm(b_rwrist - a_torso_pos, dim=-1) < self.contact_thresh
    b_hit_left = b_contact_left & (b_punch_left > self.hit_vel_threshold)
    b_hit_right = b_contact_right & (b_punch_right > self.hit_vel_threshold)
    got_hit = b_hit_left | b_hit_right
    r_defense_pen = torch.where(got_hit, torch.full_like(dmg, self.w_defense_pen), torch.zeros_like(dmg))
    # v4: hits scale with punch SPEED (not flat) so collapse-brushes (~0.5 m/s)
    # pay a fraction of a real strike (~2 m/s).
    speed_scale = torch.clamp(best_punch / self.punch_speed_ref, max=1.0)
    dmg = torch.where(
      hit, torch.clamp(best_punch * 5.0, max=20.0) * upright_scale, torch.zeros_like(rel_vel)
    )
    r_hit = torch.where(hit, torch.full_like(dmg, self.w_hit) * speed_scale * upright_scale, torch.zeros_like(dmg))

    # Fall penalty.
    fall_pen = torch.where(
      a_pelvis_z < self.fall_threshold_z,
      self.w_fall_pen * (self.fall_threshold_z - a_pelvis_z),
      torch.zeros_like(a_pelvis_z),
    )

    # HP.
    self._hp_b = torch.clamp(self._hp_b - dmg, min=0.0)

    # v5: no stand-ups. Falls are no longer teleported away — a downed robot
    # stays down until it gets up itself or the episode ends. Falling costs
    # ground time (per-step penalty), not a flat hit.
    b_pelvis_z = b_root_pos[:, 2] - env.scene.env_origins[:, 2]
    a_down = a_pelvis_z < self.knockdown_z
    b_down = b_pelvis_z < self.knockdown_z

    # Edge detection for knockdown metrics + recovery bonus.
    if not hasattr(self, "_a_was_down"):
      self._a_was_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      self._b_was_down = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      self._a_down_steps = torch.zeros(env.num_envs, device=env.device)
    a_kd_event = a_down & ~self._a_was_down
    b_kd_event = b_down & ~self._b_was_down
    self._knockdowns += (a_kd_event | b_kd_event).float()

    # Ground-time penalty: continuous bleed while A is down (~-15/s at 30 Hz).
    r_ground_pen = torch.where(a_down, torch.full_like(dmg, self.w_ground_pen), torch.zeros_like(dmg))
    self._a_down_steps = torch.where(a_down, self._a_down_steps + 1, torch.zeros_like(self._a_down_steps))

    # Recovery bonus: +recovery_credit when A returns upright after a REAL
    # knockdown (down >= 30 steps = 1 s). Gated so dip-farming can't profit
    # (30 steps of ground penalty = -15 > +5 bonus).
    recovered = (~a_down) & self._a_was_down & (self._a_down_steps >= 30) & (a_pelvis_z > 0.55)
    r_recovery = torch.where(recovered, torch.full_like(dmg, self.recovery_credit), torch.zeros_like(dmg))

    # B down WHILE A stays up: bonus (a real knockdown scored standing).
    r_sandbag_kd = torch.where(
      b_kd_event & (upright_scale > 0.5),
      torch.full_like(dmg, self.sandbag_kd_credit),
      torch.zeros_like(dmg),
    )

    self._a_was_down = a_down.clone()
    self._b_was_down = b_down.clone()

    # v2: no positive shaping while down — face/vel/dist/reach all zero below
    # the fall threshold so lying on the sandbag is strictly losing.
    up_mask = (a_pelvis_z >= self.fall_threshold_z).float()

    # v5.1: HumanUP-style get-up curriculum. When A is down (pelvis < knockdown_z),
    # combat shaping pauses and a sparse get-up reward takes over: raise pelvis
    # (height + progress), orient upright (projected gravity on z), get feet under
    # the body. Mocap-free path — RL discovers the get-up (HumanUP arXiv 2502.12152).
    a_down_now = (a_pelvis_z < self.knockdown_z).float()

    height_norm = torch.clamp(a_pelvis_z / 0.8, 0.0, 1.0)
    if not hasattr(self, "_za_prev"):
      self._za_prev = a_pelvis_z.clone()
    height_delta = (a_pelvis_z - self._za_prev).clamp(-1.0, 1.0)
    self._za_prev = a_pelvis_z.clone()
    # uprightness: projected gravity z; -1 = upright in mjlab convention, so -g_z -> 1 upright.
    uprightness = (-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    l_foot_z = robot.data.body_link_pos_w[:, self._l_foot_id, 2]
    r_foot_z = robot.data.body_link_pos_w[:, self._r_foot_id, 2]
    feet_below = ((l_foot_z < a_pelvis_z) & (r_foot_z < a_pelvis_z)).float()

    r_getup = (
        self.w_getup_height * height_norm
        + self.w_getup_progress * height_delta
        + self.w_getup_upright * (uprightness * 0.5 + 0.5)
        + self.w_getup_feet * feet_below
    ) * a_down_now

    combat_scale = 1.0 - a_down_now

    r_task = (
      self.w_face * r_face * up_mask * combat_scale
      + self.w_vel * r_vel * up_mask * combat_scale
      + self.w_dist * r_dist * up_mask * combat_scale
      + r_reach * up_mask * combat_scale
      + r_hit
      - r_defense_pen
      + r_getup
      + r_recovery
      - fall_pen
      - r_ground_pen
    )

    # AMP style reward (frozen scorer).
    r_style = torch.zeros(env.num_envs, device=env.device)
    if self._amp_disc is not None:
      jp_std = robot.data.joint_pos[:, idx.controlled_joint_indices][:, idx.robot_to_std_perm][:, idx.amp_major_std_indices]
      jv_std = robot.data.joint_vel[:, idx.controlled_joint_indices][:, idx.robot_to_std_perm][:, idx.amp_major_std_indices]
      base_ang_vel_b = _transform_to_ego(rot_mat, robot.data.root_link_ang_vel_w)
      disc_obs = torch.cat([jp_std, jv_std, base_ang_vel_b], dim=-1)
      if self._amp_prev_obs_valid.any():
        trans = torch.cat([self._amp_prev_obs, disc_obs], dim=-1)
        r_style_all = self._amp_disc.compute_reward_batch(trans)
        r_style = torch.where(self._amp_prev_obs_valid, r_style_all, torch.zeros_like(r_style_all))
      self._amp_prev_obs = disc_obs.clone()
      self._amp_prev_obs_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    total = r_task + self.amp_weight * r_style

    # Metrics -> extras["log"] (picked up by the rsl_rl logger).
    log = env.extras.setdefault("log", {})
    log["Episode/hit_rate"] = hit.float().mean().item()
    log["Episode/dmg_dealt"] = dmg.mean().item()
    log["Episode/r_face"] = r_face.mean().item()
    log["Episode/r_vel"] = r_vel.mean().item()
    log["Episode/r_dist"] = r_dist.mean().item()
    log["Episode/r_reach"] = r_reach.mean().item()
    log["Episode/r_hit"] = r_hit.mean().item()
    log["Episode/r_style"] = r_style.mean().item()
    log["Episode/dist_to_opp"] = dist.mean().item()
    log["Episode/fist_dist"] = torch.minimum(fist_dist_left, fist_dist_right).mean().item()
    log["Episode/pelvis_z"] = a_pelvis_z.mean().item()
    log["Episode/punch_speed"] = best_punch.mean().item()
    log["Episode/upright_scale"] = upright_scale.mean().item()
    log["Episode/knockdowns"] = self._knockdowns.mean().item()
    log["Episode/r_ground_pen"] = r_ground_pen.mean().item()
    log["Episode/r_recovery"] = r_recovery.mean().item()
    log["Episode/a_down_steps"] = self._a_down_steps.mean().item()
    log["Episode/r_sandbag_kd"] = r_sandbag_kd.mean().item()
    log["Episode/a_downs"] = a_kd_event.float().mean().item()
    log["Episode/b_downs_upright"] = (b_kd_event & (upright_scale > 0.5)).float().mean().item()
    log["Episode/hp_b"] = self._hp_b.mean().item()

    # Compensate for policy-dt reward scaling (see module docstring).
    return total / DECIMATION


# --------------------------------------------------------------------------- #
# Terminations
# --------------------------------------------------------------------------- #


def pelvis_below_threshold(env, entity_name: str = "robot", threshold: float = 0.4) -> torch.Tensor:
  asset: Entity = env.scene[entity_name]
  pelvis_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return pelvis_z < threshold


# --------------------------------------------------------------------------- #
# Reset event: two robots 0.6m apart, facing each other, standing pose
# --------------------------------------------------------------------------- #


def combat_reset(
  env,
  env_ids: torch.Tensor | None,
  sandbag_offset: float = 0.6,
  spawn_height: float = 0.75,
  robot_name: str = "robot",
  sandbag_name: str = "sandbag",
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  idx = _get_idx(env, robot_name, sandbag_name)
  robot: Entity = env.scene[robot_name]
  sandbag: Entity = env.scene[sandbag_name]
  device = env.device
  origins = env.scene.env_origins[env_ids]

  # Agent A at -offset/2 along x, identity quat (faces +X, toward B).
  pose_a = torch.zeros(len(env_ids), 7, device=device)
  pose_a[:, 0:3] = origins
  pose_a[:, 0] -= sandbag_offset / 2.0
  pose_a[:, 2] = spawn_height
  pose_a[:, 3] = 1.0
  vel_a = torch.zeros(len(env_ids), 6, device=device)

  jp_a = robot.data.default_joint_pos[env_ids].clone()
  jv_a = torch.zeros_like(jp_a)
  jp_a[:, idx.controlled_joint_indices] = idx.default_pose_29.unsqueeze(0)
  if idx.wrist_joint_indices is not None:
    jp_a[:, idx.wrist_joint_indices] = 0.0

  robot.write_root_link_pose_to_sim(pose_a, env_ids)
  robot.write_root_link_velocity_to_sim(vel_a, env_ids)
  robot.write_joint_state_to_sim(jp_a, jv_a, None, env_ids)

  # Agent B at +offset/2 along x, quat (0,0,0,1): 180 deg about Z (faces -X).
  pose_b = torch.zeros(len(env_ids), 7, device=device)
  pose_b[:, 0:3] = origins
  pose_b[:, 0] += sandbag_offset / 2.0
  pose_b[:, 2] = spawn_height
  pose_b[:, 6] = 1.0
  vel_b = torch.zeros(len(env_ids), 6, device=device)

  jp_b = sandbag.data.default_joint_pos[env_ids].clone()
  jv_b = torch.zeros_like(jp_b)
  jp_b[:, idx.sandbag_controlled_indices] = idx.default_pose_29.unsqueeze(0)
  if idx.sandbag_wrist_indices is not None:
    jp_b[:, idx.sandbag_wrist_indices] = 0.0

  sandbag.write_root_link_pose_to_sim(pose_b, env_ids)
  sandbag.write_root_link_velocity_to_sim(vel_b, env_ids)
  sandbag.write_joint_state_to_sim(jp_b, jv_b, None, env_ids)
