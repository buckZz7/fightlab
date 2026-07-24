"""Frozen whole-body G1 balance base: unitree_rl_mjlab velocity policy (ONNX).

Unlike the 12-DoF legs policy (motion.pt, arms pinned in training), this
29-DoF policy was trained WITH arm motion — so our fight policy can drive
the upper body without breaking it (verified: stands 30s in MuJoCo).

Interface:
  obs (98): ang_vel(3) + proj_gravity(3) + cmd(3) + gait sin/cos(2)
            + joint_pos_rel(29) + joint_vel_rel(29) + last_action(29)
  action (29): joint position targets; target = action*SCALE + HOME
  50 Hz policy, PD torque at the physics rate with deploy.yaml gains.

FightLab usage: freeze this as the balance substrate. The fight policy can
either (a) write residuals on the arm entries of the target vector, or
(b) command velocity/gait for footwork while this keeps the robot upright.
"""
import numpy as np
import onnxruntime as ort

ONNX_PATH = "/opt/data/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"

KP = np.array([40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
               40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
               40.2, 28.5, 28.5,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8])
KD = np.array([2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
               2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
               2.6, 1.8, 1.8,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1])
HOME = np.array([-0.1, 0, 0, 0.3, -0.2, 0,
                 -0.1, 0, 0, 0.3, -0.2, 0,
                 0, 0, 0,
                 0.35, 0.18, 0, 0.87, 0, 0, 0,
                 0.35, -0.18, 0, 0.87, 0, 0, 0])
SCALE = np.array([0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                  0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
                  0.55, 0.44, 0.44,
                  0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,
                  0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07])
GAIT_PERIOD = 0.6
DECIMATION = 10           # 50 Hz at dt=0.002

# Shared ONNX session: inference is stateless given obs, and ort releases the
# GIL during run() — one session serves all env instances/threads. Cuts
# per-env memory by the session+model size.
_SHARED_SESSION = None


def _get_session(path=ONNX_PATH):
    global _SHARED_SESSION
    if _SHARED_SESSION is None:
        _SHARED_SESSION = ort.InferenceSession(path)
    return _SHARED_SESSION

# Joint index groups (29-DoF actuator order: legs 0-11, waist 12-14,
# left arm 15-21, right arm 22-28)
LEGS = slice(0, 12)
WAIST = slice(12, 15)
ARMS = slice(15, 29)
L_ARM = slice(15, 22)
R_ARM = slice(22, 29)


class LocoBase29:
    def __init__(self, onnx_path=ONNX_PATH):
        self.sess = _get_session(onnx_path)
        self.inp = self.sess.get_inputs()[0].name
        self.action = np.zeros(29, dtype=np.float32)
        self.target = HOME.copy()
        self.cmd = np.zeros(3, dtype=np.float32)
        self._counter = 0

    def set_command(self, vx=0.0, vy=0.0, wz=0.0):
        self.cmd[:] = [vx, vy, wz]

    @staticmethod
    def _grav_ori(q):
        qw, qx, qy, qz = q
        return np.array([2 * (-qz * qx + qw * qy),
                         -2 * (qz * qy + qw * qx),
                         1 - 2 * (qw * qw + qz * qz)])

    def update(self, qpos, qvel):
        """Call once per physics step; recomputes policy at 50 Hz."""
        self._counter += 1
        if self._counter % DECIMATION != 0:
            return self.target
        phase = (self._counter * 0.002 % GAIT_PERIOD) / GAIT_PERIOD
        obs = np.concatenate([
            qvel[3:6],
            self._grav_ori(qpos[3:7]),
            self.cmd,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            qpos[7:36] - HOME,
            qvel[6:35],
            self.action,
        ]).astype(np.float32)
        self.action = self.sess.run(None, {self.inp: obs[None]})[0].squeeze()
        self.target = self.action * SCALE + HOME
        return self.target

    def pd_torque(self, qpos, qvel, target_override=None):
        """PD torques toward target (optionally overridden by fight policy)."""
        t = self.target if target_override is None else target_override
        return KP * (t - qpos[7:36]) - KD * qvel[6:35]

    def reset(self):
        self.action[:] = 0
        self.target = HOME.copy()
        self.cmd[:] = 0
        self._counter = 0
