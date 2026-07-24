"""Test unitree_rl_mjlab's whole-body 29-DoF velocity policy (policy.onnx)
standing still in MuJoCo. If this holds, it's the full-body balance base
FightLab needs (arms included in training, unlike the 12-DoF legs policy).

Obs layout (from deploy.yaml): ang_vel(3) + projected_gravity(3) + cmd(3)
  + gait_phase(2: sin/cos, period 0.6s) + joint_pos_rel(29) + joint_vel_rel(29)
  + last_action(29) = 98
Actions: joint position targets, scale per-joint + offset (deploy.yaml).
"""
import mujoco
import numpy as np
import onnxruntime as ort

XML = "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
ONNX = "/opt/data/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"

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

sess = ort.InferenceSession(ONNX)
inp = sess.get_inputs()[0].name
print("onnx input:", sess.get_inputs()[0].shape, "output:", sess.get_outputs()[0].shape)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
m.opt.timestep = 0.002
d.qpos[2] = 0.75
d.qpos[7:36] = HOME
mujoco.mj_forward(m, d)

pelvis = m.body("pelvis").id
lo = m.actuator_ctrlrange[:, 0]
hi = m.actuator_ctrlrange[:, 1]

action = np.zeros(29, dtype=np.float32)
target = HOME.copy()
cmd = np.zeros(3, dtype=np.float32)

def grav_ori(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])

z_min, z_end = 1.0, 0.0
i = 0
for i in range(15000):  # 30s
    tau = KP * (target - d.qpos[7:36]) - KD * d.qvel[6:35]
    d.ctrl[:] = np.clip(tau, lo, hi)
    mujoco.mj_step(m, d)
    if i % 10 == 0:  # 50 Hz policy
        phase = (i * 0.002 % GAIT_PERIOD) / GAIT_PERIOD
        obs = np.concatenate([
            d.qvel[3:6],                       # ang vel
            grav_ori(d.qpos[3:7]),             # projected gravity
            cmd,                               # velocity command
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            d.qpos[7:36] - HOME,               # joint pos rel
            d.qvel[6:35],                      # joint vel rel
            action,                            # last action
        ]).astype(np.float32)
        action = sess.run(None, {inp: obs[None]})[0].squeeze()
        target = action * SCALE + HOME
    z = float(d.xpos[pelvis][2])
    z_min = min(z_min, z)
    z_end = z
    if z < 0.4:
        break

print(f"mjlab whole-body policy standing: {i+1} steps ({(i+1)*0.002:.1f}s) "
      f"z min={z_min:.2f} end={z_end:.2f}")
