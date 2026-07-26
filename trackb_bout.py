"""Track B 2-bot bout: two HYBRID G1 fighters (balance + G1 Moves arms).

r1 = scripted striker (cycles jab / lowpunch / rapidpunch, with footwork).
r2 = guard holder (stance + slight footwork), also balanced.
Both use G1MovesAgent: legs/waist = LocoBase29 ONNX (balance),
arms = G1 Moves strike blend.

Proves: two full-body fighters coexist + strike in one arena, balanced.
Next: learned fight-layer policy replaces the scripted striker.

Usage:
  python trackb_bout.py --out trackb_bout.mp4 --seconds 12
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import imageio.v2 as imageio
from g1_arena import build_arena
from g1moves_agent import G1MovesAgent, SKILLS
from loco_base29 import HOME

QPOS_OFF = [0, 7]
QVEL_OFF = [0, 6]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="trackb_bout.mp4")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--motion", default=os.path.join(os.path.dirname(__file__), "g1moves"))
    ap.add_argument("--onnx", default=os.environ.get("G1_ONNX_PATH"))
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    model = build_arena()
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002  # 500 Hz

    onnx = a.onnx or os.environ.get("G1_ONNX_PATH")
    agents = [G1MovesAgent(a.motion, onnx_path=onnx),
              G1MovesAgent(a.motion, onnx_path=onnx)]

    # init: root world pos from arena (r1 x=-0.6, r2 x=0.3, z=0.793),
    # joints from HOME (the balanced stance LocoBase29 expects)
    roots = [[-0.6, 0, 0.793], [0.3, 0, 0.793]]
    for ai, off in enumerate(QPOS_OFF):
        data.qpos[off+7 : off+7+3] = roots[ai]      # root x,y,z (world)
        data.qpos[off+7+3 : off+7+29] = HOME[3:]   # joint targets as init pose
    mujoco.mj_forward(model, data)

    rend = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]

    dt = 0.02
    steps = int(a.seconds / dt)
    frames = []
    strike_timer = 0.0
    cur_skill = -1
    for i in range(steps):
        t = i * dt
        # r1 scripted striker: cycle punch skills every 3s
        if t - strike_timer > 3.0:
            strike_timer = t
            cur_skill = (cur_skill + 1) % 3
            agents[0].command(cur_skill, (0.15, 0, 0))  # step in
        else:
            agents[0].command(cur_skill if cur_skill >= 0 else -1, (0.05, 0, 0))
        # r2 guard + slight sway
        agents[1].command(-1, (0, 0, 0))

        for ai, off in enumerate(QPOS_OFF):
            qv = QVEL_OFF[ai]
            tau = agents[ai].step(model, data, qpos_off=off, qvel_off=qv)
            base = ai * 29
            data.ctrl[base:base+29] = tau

        for _ in range(10):  # 0.02s @ 500Hz
            mujoco.mj_step(model, data)
        if i % 2 == 0:
            cam.azimuth = 90 + 18 * np.sin(i * 0.003)
            rend.update_scene(data, camera=cam)
            frames.append(rend.render())

    imageio.imsave(a.out, frames, fps=a.fps)
    print(f"TRACK B BOUT rendered {len(frames)} frames -> {a.out}")

if __name__ == "__main__":
    main()
