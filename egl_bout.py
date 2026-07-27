"""EGL 1v1 eval bout: saves frames as PNGs, then ffmpeg encodes."""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML", "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR", "/workspace/unitree_mujoco/unitree_robots/g1/meshes")

import numpy as np
import mujoco
from g1_fighter_env import G1FighterEnv
from bout_fighter import ShadowBoxer
from boxing_rules import BoxingJudge
import PIL.Image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--p1", default=None, help="fighter policy path (None=shadowboxer)")
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--out", default="/tmp/egl_bout.mp4")
ap.add_argument("--no-terminate", action="store_true")
ap.add_argument("--frames-dir", default="/tmp/bout_frames")
a = ap.parse_args()

env = G1FighterEnv(max_steps=a.steps, randomize=False, demo=(a.p1 is None))
obs, _ = env.reset()
judge = BoxingJudge(env, round_seconds=3.0, rounds=3)

if a.p1:
    from stable_baselines3 import PPO
    p1 = PPO.load(a.p1)
else:
    p1 = ShadowBoxer(env, style="red")

env.opponent = ShadowBoxer(env, style="blue")

cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
r = mujoco.Renderer(env.model, height=720, width=1280)

os.makedirs(a.frames_dir, exist_ok=True)
n_frames = 0
for t in range(a.steps):
    a1, _ = p1.predict(obs, deterministic=True)
    obs, rew, term, trunc, info = judge.step(a1)
    try:
        r.update_scene(env.data, camera=cam_id)
        img = r.render()
        # Save every frame as PNG
        PIL.Image.fromarray(img).save(f"{a.frames_dir}/f{t:05d}.png")
        n_frames += 1
    except Exception:
        pass
    if t % 1000 == 0:
        print(f"step {t}, hp={env.hp}, frames={n_frames}", flush=True)
    if not a.no_terminate and (term or trunc):
        break

print(f"total: {n_frames} frames, hp={env.hp}", flush=True)

# Encode with imageio_ffmpeg (ffmpeg not installed on pod)
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-framerate", "30", "-i", f"{a.frames_dir}/f%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
               capture_output=True)
os.system(f"rm -rf {a.frames_dir}")
print(f"SAVED {a.out}", flush=True)
card = judge.card()
print(f"CARD: winner={card['winner']} method={card['method']} hp={card['final_hp']}", flush=True)
