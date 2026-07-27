"""Eval the motion tracker: load the trained policy, run it on a
karate motion, render to see if the G1 actually moves."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import numpy as np
import mujoco
import PIL.Image
import imageio_ffmpeg
import subprocess
from stable_baselines3 import PPO
from train_motion_tracker import load_motions, MotionTrackerEnv

motions = load_motions("/workspace/g1-moves/karate")
model = PPO.load("/workspace/models/motion_tracker.zip")

# Pick the attack motion for the eval
attack_motions = [m for m in motions if "Attack" in m["name"]]
if not attack_motions:
    attack_motions = motions
motion_idx = 0  # first attack motion

env = MotionTrackerEnv(attack_motions, max_steps=500)
# Force the same motion each time for consistent eval
env.motions = [attack_motions[motion_idx]]
obs, _ = env.reset()
print(f"eval motion: {attack_motions[motion_idx]['name']}, {len(attack_motions[motion_idx]['joint_pos'])} frames")

cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
r = mujoco.Renderer(env.model, height=720, width=1280)

frames_dir = "/tmp/tracker_eval_frames"
os.makedirs(frames_dir, exist_ok=True)
n = 0
total_reward = 0
for t in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    try:
        r.update_scene(env.data, camera=cam_id)
        img = r.render()
        PIL.Image.fromarray(img).save(f"{frames_dir}/f{t:05d}.png")
        n += 1
    except:
        pass
    if t % 100 == 0:
        pelvis_z = env.data.xpos[env.model.body("r1_pelvis").id][2]
        print(f"step {t}: reward={reward:.3f} pelvis_z={pelvis_z:.3f}", flush=True)
    if term or trunc:
        break

print(f"total: {n} frames, avg_reward={total_reward/max(n,1):.3f}", flush=True)

# Encode
out = "/tmp/tracker_eval.mp4"
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-framerate", "30", "-i", f"{frames_dir}/f%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
               capture_output=True)
os.system(f"rm -rf {frames_dir}")
print(f"SAVED {out}", flush=True)
