"""Track B 2-bot bout: two full-body G1 Moves agents in the arena.

r1 = scripted striker (throws jab / low punch on a timer).
r2 = stance holder (guard, slight movement).
Both controlled by G1MovesAgent PD (stable in our sim).

Proves: full-body priors drive BOTH robots in one arena, strikes execute.
Next phase: replace the scripted striker with a learned fight-layer policy.

Usage:
  python trackb_bout.py --out trackb_bout.mp4 --seconds 20
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import imageio.v2 as imageio
from g1_arena import build_arena
from g1moves_agent import G1MovesAgent, SKILLS

QPOS_OFF = [0, 7]   # r1 at 0, r2 at +7
QVEL_OFF = [0, 6]
NQ = 29

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="trackb_bout.mp4")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--motion", default=os.path.join(os.path.dirname(__file__), "g1moves"))
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    model = build_arena()
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005  # 200 Hz

    agents = [G1MovesAgent(a.motion, "x"), G1MovesAgent(a.motion, "x")]

    # init both from stance (r1 jab frame 0, r2 stance)
    for ai, off in enumerate(QPOS_OFF):
        jp = agents[ai].clips.get("jab", {}).get("jp")
        if jp is not None:
            data.qpos[off+7 : off+7+29] = jp[0]
        else:
            data.qpos[off+7 : off+7+29] = agents[ai].target
    # separate the two robots in X
    data.qpos[QPOS_OFF[1] + 0] = 0.9   # r2 forward
    mujoco.mj_forward(model, data)

    rend = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]

    dt = 0.02  # control at 50Hz
    steps = int(a.seconds / dt)
    frames = []
    strike_timer = 0.0
    cur_skill = -1
    for i in range(steps):
        t = i * dt
        # scripted striker: cycle jab -> lowpunch -> rapidpunch every 3s
        if t - strike_timer > 3.0:
            strike_timer = t
            cur_skill = (cur_skill + 1) % 3  # jab/lowpunch/rapidpunch
            agents[0].command(cur_skill, (0, 0, 0))
        else:
            if cur_skill == -1:
                agents[0].command(-1, (0, 0, 0))
        # r2 holds guard, slight sway
        agents[1].command(-1, (0, 0, 0))

        # PD for both
        for ai, off in enumerate(QPOS_OFF):
            qv_off = QVEL_OFF[ai]
            tau = agents[ai].step(model, data, qpos_off=off, qvel_off=qv_off)
            base = ai * 29
            data.ctrl[base:base+29] = tau

        for _ in range(4):
            mujoco.mj_step(model, data)
        if i % 2 == 0:
            cam.azimuth = 90 + 18 * np.sin(i * 0.003)
            rend.update_scene(data, camera=cam)
            frames.append(rend.render())

    imageio.mimsave(a.out, frames, fps=a.fps)
    print(f"TRACK B BOUT rendered {len(frames)} frames -> {a.out}")

if __name__ == "__main__":
    main()
