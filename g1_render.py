"""G1 stick-figure fight renderer (headless — no GL needed).

Renders a G1 bout/punch sequence as 2D stick figures from kinematic data:
side view (x-z plane), full skeleton, fists highlighted, bag with swing,
HUD overlay (bag velocity, pelvis height, time).

Usage:
  from g1_render import G1Renderer
  r = G1Renderer(model)
  r.frame(data, t=i*0.002, hud={"bag_vel": v})   # per frame
  r.save("fight.mp4")

Or render a trained punch policy rollout:
  python g1_render.py --model models/g1_punch_ppo.zip --out punch.mp4
"""
import argparse

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# G1 29dof kinematic chains (body names) for stick drawing
CHAINS = {
    "torso": ["pelvis", "torso_link", "head_link"],
    "l_arm": ["torso_link", "left_shoulder_pitch_link", "left_elbow_link",
              "left_wrist_yaw_link"],
    "r_arm": ["torso_link", "right_shoulder_pitch_link", "right_elbow_link",
              "right_wrist_yaw_link"],
    "l_leg": ["pelvis", "left_hip_pitch_link", "left_knee_link",
              "left_ankle_roll_link"],
    "r_leg": ["pelvis", "right_hip_pitch_link", "right_knee_link",
              "right_ankle_roll_link"],
}
CHAIN_STYLE = {
    "torso": dict(c="#f2f0eb", lw=3),
    "l_arm": dict(c="#8ab4f8", lw=2),
    "r_arm": dict(c="#e8291c", lw=2),   # red = punching arm
    "l_leg": dict(c="#6b6b6b", lw=2.5),
    "r_leg": dict(c="#9a9a9a", lw=2.5),
}


class G1Renderer:
    def __init__(self, model, width=640, height=480, dpi=80):
        self.model = model
        self._body_ids = {}
        for chain, names in CHAINS.items():
            ids = []
            for n in names:
                try:
                    ids.append(self.model.body(n).id)
                except KeyError:
                    pass
            self._body_ids[chain] = ids
        self._fist_ids = []
        for side in ("left", "right"):
            try:
                self._fist_ids.append(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                      f"{side}_fist_col"))
            except Exception:
                pass
        self._bag_body = None
        try:
            self._bag_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "heavy_bag")
        except Exception:
            pass

        self.fig, self.ax = plt.subplots(
            figsize=(width / dpi, height / dpi), dpi=dpi)
        self.frames = []

    def frame(self, data, t=0.0, hud=None):
        self.ax.clear()
        ax = self.ax
        ax.set_facecolor("#0a0a0a")
        self.fig.patch.set_facecolor("#0a0a0a")

        for chain, ids in self._body_ids.items():
            if not ids:
                continue
            xs = [float(data.xpos[b][0]) for b in ids]
            zs = [float(data.xpos[b][2]) for b in ids]
            ax.plot(xs, zs, "-o", ms=3, **CHAIN_STYLE[chain])

        # fists
        for gid in self._fist_ids:
            p = data.geom_xpos[gid]
            ax.scatter([p[0]], [p[2]], c="#e8291c", s=90, marker="o",
                       edgecolors="white", linewidths=1.2, zorder=5)

        # bag
        if self._bag_body is not None:
            bp = data.xpos[self._bag_body]
            bag = plt.Circle((bp[0], bp[2]), 0.12, color="#e8291c", alpha=0.85)
            ax.add_patch(bag)
            ax.plot([0.30, bp[0]], [0.0, bp[2]], "-", c="#555555", lw=1.5)

        ax.axhline(0, c="#f2f0eb", lw=2)
        ax.set_xlim(-0.6, 1.2)
        ax.set_ylim(-0.1, 1.9)
        ax.set_aspect("equal")
        ax.axis("off")

        hud = hud or {}
        lines = [f"t={t:.1f}s"]
        if "bag_vel" in hud:
            lines.append(f"bag v={hud['bag_vel']:.2f} m/s")
        if "pelvis_z" in hud:
            lines.append(f"z={hud['pelvis_z']:.2f}")
        ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
                va="top", color="#f2f0eb", fontsize=9, family="monospace")

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        self.frames.append(buf)

    def save(self, path, fps=30):
        imageio.mimsave(path, self.frames, fps=fps)
        return {"video": path, "frames": len(self.frames)}


def render_punch_rollout(model_path, out="punch.mp4", seconds=10, seed=7):
    """Roll out a trained G1PunchEnv policy and render it."""
    import sys
    sys.path.insert(0, "/opt/data/fightlab-repo-new")
    from stable_baselines3 import PPO
    from g1_punch_env import G1PunchEnv

    env = G1PunchEnv(randomize=False)
    obs, _ = env.reset(seed=seed)
    model = PPO.load(model_path)
    r = G1Renderer(env.model)
    fps = 30
    every = max(1, int((1.0 / fps) / 0.02))   # env runs at 50 Hz control
    steps = int(seconds / 0.02)
    for i in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(a)
        if i % every == 0:
            r.frame(env.data, t=i * 0.02,
                    hud={"bag_vel": info["bag_vel"], "pelvis_z": info["pelvis_z"]})
        if term or trunc:
            obs, _ = env.reset(seed=seed + 1)
    result = r.save(out, fps=fps)
    print(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="punch.mp4")
    ap.add_argument("--seconds", type=float, default=10)
    args = ap.parse_args()
    render_punch_rollout(args.model, args.out, args.seconds)
