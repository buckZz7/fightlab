"""Test damage detection: place two bots, force a punch, check HP drops."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import numpy as np
import mujoco
from g1_fighter_env import G1FighterEnv

env = G1FighterEnv(max_steps=200, randomize=False)
obs, _ = env.reset()

# Move bots point-blank — 0.1m apart
env.data.qpos[0:3] = [-0.05, 0, 0.793]   # r1
env.data.qpos[36:39] = [0.05, 0, 0.793]  # r2 (0.1m apart)
mujoco.mj_forward(env.model, env.data)

print(f"Initial HP: {env.hp}")
print(f"Weapons r1: {env.fist_geoms[0]}")
print(f"Torso bodies r2: {env.torso_bodies[1]}")

# Apply a "punch" — extend r1's right arm toward r2
# Right arm joints: indices 22-28 in qpos (7+15=22 to 7+29=36)
# Right shoulder pitch (index 22): forward swing
# Right elbow (index 25): extend
# Right wrist (26-28): straight
home = env.native.copy()

hits = 0
for t in range(200):
    # Override r1's right arm to punch toward r2
    # shoulder_pitch forward = negative value
    # elbow extend = positive value
    punch_target = home.copy()
    punch_target[15] = -0.8   # right shoulder pitch (forward)
    punch_target[18] = 1.5    # right elbow (extend)

    # Apply PD to r1
    kp, kd = 100.0, 5.0
    tau1 = kp * (punch_target - env.data.qpos[7:36]) - kd * env.data.qvel[6:35]
    env.data.ctrl[0:29] = np.clip(tau1, -88, 88)

    # r2 just stands
    tau2 = kp * (home - env.data.qpos[43:72]) - kd * env.data.qvel[41:70]
    env.data.ctrl[29:58] = np.clip(tau2, -88, 88)

    mujoco.mj_step(env.model, env.data, 1)
    env._update_damage()

    if env._dmg_dealt[0] > 0:
        hits += 1
        if hits <= 5:
            print(f"step {t}: HIT! dmg={env._dmg_dealt[0]:.2f} hp={env.hp}")

    if t % 50 == 0:
        # Check wrist position relative to r2 torso
        r1_wrist = env.data.xpos[env.model.body("r1_right_wrist_yaw_link").id]
        r2_torso = env.data.xpos[env.model.body("r2_torso_link").id]
        dist = np.linalg.norm(r1_wrist - r2_torso)
        print(f"step {t}: wrist-torso dist={dist:.3f} hp={env.hp}")

    if env.hp[1] < 50:
        print(f"R2 HP below 50 at step {t}!")
        break

print(f"\nFinal HP: {env.hp}")
print(f"Total hits landed: {hits}")
if hits == 0:
    print("NO DAMAGE DETECTED — collision/damage system needs fixing")
else:
    print("Damage working!")
