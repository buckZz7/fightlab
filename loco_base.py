"""Frozen G1 locomotion base: unitree_rl_gym's pretrained 12-DoF legs policy.

Wraps motion.pt (torch.jit) as the lower-body controller for FightLab envs:
  - Policy runs at 50 Hz (every 10 physics steps at 0.002s)
  - PD torque at the physics rate: tau = kp*(target-q) - kd*qd
  - Velocity command interface: cmd=(vx, vy, wz); hold (0,0,0) to stand

Verified: stands 30s+ in stock MuJoCo scene (g1test_baseline.py).

This is the proven foundation — the fight layer builds on top of it.
"""
import numpy as np
import torch

POLICY_PATH = "/opt/data/unitree_rl_gym/deploy/pre_train/g1/motion.pt"

# 12 leg joints, order matches g1_12dof / first 12 actuators of the 29dof XML
LEG_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float64)
LEG_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float64)
LEG_HOME = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, dtype=np.float64)

ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25])
NUM_ACTIONS = 12
NUM_OBS = 47
DECIMATION = 10            # policy every 10 physics steps -> 50 Hz
GAIT_PERIOD = 0.8          # seconds, from their deploy code


class LocomotionBase:
    """Frozen legs policy: obs in, leg position targets out."""

    def __init__(self, policy_path=POLICY_PATH):
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        self.action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.leg_target = LEG_HOME.copy()
        self.cmd = np.zeros(3, dtype=np.float32)
        self._obs = np.zeros(NUM_OBS, dtype=np.float32)
        self._counter = 0

    def set_command(self, vx=0.0, vy=0.0, wz=0.0):
        self.cmd[:] = [vx, vy, wz]

    @staticmethod
    def _gravity_orientation(q):
        qw, qx, qy, qz = q
        return np.array([2 * (-qz * qx + qw * qy),
                         -2 * (qz * qy + qw * qx),
                         1 - 2 * (qw * qw + qz * qz)])

    def update(self, qpos, qvel):
        """Advance the policy counter; returns leg position targets (12,).

        qpos: full model qpos (root 7 + joints). qvel: full model qvel.
        Recomputes the policy action every DECIMATION calls; otherwise holds
        the previous target (call once per physics step).
        """
        self._counter += 1
        if self._counter % DECIMATION != 0:
            return self.leg_target

        leg_q = qpos[7:7 + NUM_ACTIONS]
        leg_qd = qvel[6:6 + NUM_ACTIONS]
        quat = qpos[3:7]
        omega = qvel[3:6]

        o = self._obs
        o[:3] = omega * ANG_VEL_SCALE
        o[3:6] = self._gravity_orientation(quat)
        o[6:9] = self.cmd * CMD_SCALE
        o[9:21] = (leg_q - LEG_HOME)          # dof_pos_scale = 1.0
        o[21:33] = leg_qd * DOF_VEL_SCALE
        o[33:45] = self.action
        phase = (self._counter * 0.002 % GAIT_PERIOD) / GAIT_PERIOD
        o[45] = np.sin(2 * np.pi * phase)
        o[46] = np.cos(2 * np.pi * phase)

        with torch.no_grad():
            self.action = self.policy(
                torch.from_numpy(o).unsqueeze(0)).numpy().squeeze()
        self.leg_target = self.action * ACTION_SCALE + LEG_HOME
        return self.leg_target

    def pd_torque(self, qpos, qvel):
        """Leg PD torques for the current targets (call every physics step)."""
        leg_q = qpos[7:7 + NUM_ACTIONS]
        leg_qd = qvel[6:6 + NUM_ACTIONS]
        return (self.leg_target - leg_q) * LEG_KP - leg_qd * LEG_KD

    def reset(self):
        self.action[:] = 0
        self.leg_target = LEG_HOME.copy()
        self.cmd[:] = 0
        self._counter = 0
