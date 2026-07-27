"""Render league bouts to MP4s using EGL GPU rendering + tracking camera.

Reads league_standings.json, renders the top matchups to docs/bouts/*.mp4.
720p, broadcast tracking camera, navy checkerboard, fast GPU render.

Usage:
  python3 render_league_bouts.py --standings docs/league_standings.json \
      --pd --steps 5000 --out-dir docs/bouts
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import numpy as np
import mujoco
from stable_baselines3 import PPO
import PIL.Image
import imageio_ffmpeg
import subprocess

from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge
from league import _load_entrant


def render_bout(spec_a, spec_b, balance, out, steps):
    """Render a single bout to MP4 using EGL + tracking camera."""
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=steps, randomize=False)
    env.opponent = _load_entrant(spec_b, env, for_blue=True)
    judge = BoxingJudge(env, round_seconds=30.0, rounds=3)
    red = _load_entrant(spec_a, env, for_blue=False)

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
    r = mujoco.Renderer(env.model, height=720, width=1280)

    frames_dir = out.replace(".mp4", "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    obs, _ = env.reset()
    n = 0
    for t in range(steps):
        a1, _ = red.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)
        try:
            r.update_scene(env.data, camera=cam_id)
            img = r.render()
            PIL.Image.fromarray(img).save(f"{frames_dir}/f{t:05d}.png")
            n += 1
        except Exception:
            pass
        if term or trunc:
            break

    # Encode with ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", "30", "-i",
                    f"{frames_dir}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", out],
                   capture_output=True)
    # Clean up frames
    os.system(f"rm -rf {frames_dir}")
    print(f"[render] {out} ({n} frames) hp={env.hp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out-dir", default="docs/bouts")
    ap.add_argument("--max-bouts", type=int, default=4)
    ap.add_argument("--pd", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    standings = json.load(open(a.standings))
    bouts = standings.get("bouts", [])[:a.max_bouts]

    for bout in bouts:
        name_a = bout["a"]
        name_b = bout["b"]
        spec_a = bout.get("spec_a", name_a)
        spec_b = bout.get("spec_b", name_b)
        fname = f"{name_a}_vs_{name_b}".replace(" ", "_").replace(":", "_")
        out = os.path.join(a.out_dir, f"{fname}.mp4")
        print(f"[render] {name_a} vs {name_b} -> {out}")
        try:
            render_bout(spec_a, spec_b, None, out, a.steps)
        except Exception as e:
            print(f"  [error] {e}")


if __name__ == "__main__":
    main()
