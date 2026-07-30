"""Unitree G1 FightLab combat warmup environment configuration (mjlab port).

Two G1 29-DoF robots: trainable "robot" + frozen "sandbag" holding a standing
pose. 32D residual-latent action through a frozen decoder+prior.
"""

from dataclasses import replace
import os

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.combat import mdp as combat_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# PD gains from the Isaac IdealPD actuator config (warmup_env_cfg.py).
_KP_KD = {
  "hip_pitch": (40.179, 2.558),
  "hip_yaw": (40.179, 2.558),
  "waist_yaw": (40.179, 2.558),
  "hip_roll_knee": (99.098, 6.309),
  "ankle": (28.501, 1.814),
  "waist_roll_pitch": (28.501, 1.814),
  "arm": (14.251, 0.907),  # shoulders, elbows, wrist_roll
  "wrist_pitch_yaw": (16.778, 1.068),
}


def _isaac_pd_articulation() -> EntityArticulationInfoCfg:
  """G1 articulation with Isaac IdealPD kp/kd overrides."""
  kp_hp, kd_hp = _KP_KD["hip_pitch"]
  kp_hk, kd_hk = _KP_KD["hip_roll_knee"]
  kp_an, kd_an = _KP_KD["ankle"]
  kp_wr, kd_wr = _KP_KD["waist_roll_pitch"]
  kp_ar, kd_ar = _KP_KD["arm"]
  kp_wp, kd_wp = _KP_KD["wrist_pitch_yaw"]
  return EntityArticulationInfoCfg(
    actuators=(
      BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
        stiffness=kp_hp,
        damping=kd_hp,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
        stiffness=kp_hk,
        damping=kd_hk,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        stiffness=kp_an,
        damping=kd_an,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
        stiffness=kp_wr,
        damping=kd_wr,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(
          ".*_shoulder_pitch_joint",
          ".*_shoulder_roll_joint",
          ".*_shoulder_yaw_joint",
          ".*_elbow_joint",
          ".*_wrist_roll_joint",
        ),
        stiffness=kp_ar,
        damping=kd_ar,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
        stiffness=kp_wp,
        damping=kd_wp,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )


def _combat_g1_cfg():
  cfg = get_g1_robot_cfg()
  return replace(cfg, articulation=_isaac_pd_articulation())


def unitree_g1_combat_warmup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Two-G1 combat warmup env: agent vs frozen sandbag."""

  observations = {
    "actor": ObservationGroupCfg(
      terms={"combat_obs": ObservationTermCfg(func=combat_mdp.combat_obs)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={"combat_obs": ObservationTermCfg(func=combat_mdp.combat_obs)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions = {
    "latent_residual": combat_mdp.LatentResidualActionCfg(
      entity_name="robot",
      latent_ckpt_path="/workspace/latent_29/latent_29.pt",
      prior_scale=0.5,
      action_scale=0.25,
      sandbag_name="sandbag",
      # Self-play: frozen opponent policy drives robot B (empty = sandbag pose-hold).
      opponent_ckpt_path=os.environ.get("FIGHTLAB_OPPONENT_CKPT", ""),
    )
  }

  events = {
    "combat_reset": EventTermCfg(
      func=combat_mdp.combat_reset,
      mode="reset",
      params={
        "sandbag_offset": 0.6,
        "spawn_height": 0.75,
        "robot_name": "robot",
        "sandbag_name": "sandbag",
      },
    ),
    # Random shoves so fighters learn to take a push (combat robustness).
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.3, 0.3),
          "roll": (-0.4, 0.4),
          "pitch": (-0.4, 0.4),
          "yaw": (-0.6, 0.6),
        },
      },
    ),
  }

  rewards = {
    "combat": RewardTermCfg(
      func=combat_mdp.CombatReward,
      weight=1.0,
      params={
        "sigma_face": 0.3,
        "sigma_vel": 0.5,
        "sigma_dist": 0.5,
        "w_face": 1.0,
        "w_vel": 2.0,
        "w_dist": 2.0,
        "w_hit": 10.0,
        "w_fall_pen": 50.0,
        "hit_vel_threshold": 0.5,
        "contact_thresh": 0.35,
        "fall_threshold_z": 0.5,
        "knockdown_z": 0.4,
        "sandbag_offset": 0.6,
        "w_knockdown_pen": 50.0,
        "punch_speed_ref": 2.0,
        "reach_weight": 0.1,
        "sandbag_kd_credit": 15.0,
        "max_hp": 100.0,
        "amp_weight": 0.2,
        "amp_ckpt": "/workspace/amp_pretrained_punching.pt",
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=200,
      njmax=900,
      mujoco=MujocoCfg(
        timestep=1.0 / 120.0,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=30.0,
  )

  cfg.scene.entities = {"robot": _combat_g1_cfg(), "sandbag": _combat_g1_cfg()}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.terminations.pop("robot_fell", None)
    cfg.terminations.pop("sandbag_fell", None)

  return cfg
