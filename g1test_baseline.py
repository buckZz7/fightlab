"""Run unitree_rl_gym's pretrained G1 locomotion policy (motion.pt) in MuJoCo,
commanded to STAND STILL. Establishes the known-good baseline: their model +
their policy + their PD gains."""
import mujoco, numpy as np, torch

m = mujoco.MjModel.from_xml_path("/opt/data/unitree_rl_gym/resources/robots/g1_description/scene.xml")
d = mujoco.MjData(m)
m.opt.timestep = 0.002
kps = np.array([100,100,100,150,40,40]*2, dtype=np.float32)
kds = np.array([2,2,2,4,2,2]*2, dtype=np.float32)
default_angles = np.array([-0.1,0,0,0.3,-0.2,0]*2, dtype=np.float32)
policy = torch.jit.load("/opt/data/unitree_rl_gym/deploy/pre_train/g1/motion.pt")
pelvis = m.body("pelvis").id

def grav_ori(q):
    qw,qx,qy,qz = q
    return np.array([2*(-qz*qx+qw*qy), -2*(qz*qy+qw*qx), 1-2*(qw*qw+qz*qz)])

action = np.zeros(12, dtype=np.float32)
target = default_angles.copy()
obs = np.zeros(47, dtype=np.float32)
cmd = np.array([0.,0.,0.], dtype=np.float32)  # stand still
cmd_scale = np.array([2.0,2.0,0.25])
z_min, z_end = 1.0, 0.0
i = 0
for i in range(15000):  # 30s
    tau = (target - d.qpos[7:])*kps - d.qvel[6:]*kds
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    if i % 10 == 0:
        qj = (d.qpos[7:] - default_angles)
        dqj = d.qvel[6:] * 0.05
        g = grav_ori(d.qpos[3:7])
        om = d.qvel[3:6] * 0.25
        period = 0.8; phase = (i*0.002 % period)/period
        obs[:3]=om; obs[3:6]=g; obs[6:9]=cmd*cmd_scale
        obs[9:21]=qj; obs[21:33]=dqj; obs[33:45]=action
        obs[45:47]=[np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
        action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        target = action*0.25 + default_angles
    z = float(d.xpos[pelvis][2]); z_min=min(z_min,z); z_end=z
    if z < 0.4: break

print(f"THEIR POLICY standing: {i+1} steps ({(i+1)*0.002:.1f}s) z min={z_min:.2f} end={z_end:.2f}")
