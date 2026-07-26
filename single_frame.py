"""Single-frame render for fast camera/design iteration.

Renders ONE frame of the ring with the two bots, so we can
tune camera/arena params in seconds instead of waiting for a
full video render. Runs on the pod (mujoco + OSMesa).

Usage (on pod):
  python3 single_frame.py --cam_az 50 --cam_el 8 --cam_dist 5.5 \
      --cam_lookat "-0.15 0 0.85" --out /tmp/frame.png
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from g1_arena import build_arena


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_az", type=float, default=50.0)
    ap.add_argument("--cam_el", type=float, default=8.0)
    ap.add_argument("--cam_dist", type=float, default=5.5)
    ap.add_argument("--cam_lookat", default="-0.15 0 0.85")
    ap.add_argument("--out", default="/tmp/frame.png")
    ap.add_argument("--size", default="960x540")
    a = ap.parse_args()
    w, h = [int(x) for x in a.size.split("x")]

    model = build_arena(ring="ropes", half=2.4)
    data = mujoco.MjData(model)
    # place both bots upright (neutral HOME-ish stand)
    for ai, x in enumerate([-0.6, 0.3]):
        off = ai * 36
        data.qpos[off:off+3] = [x, 0, 0.76]
        data.qpos[off+3:off+7] = [1, 0, 0, 0]  # upright quat
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = a.cam_az
    cam.elevation = a.cam_el
    cam.distance = a.cam_dist
    cam.lookat[:] = [float(x) for x in a.cam_lookat.split()]

    rend = mujoco.Renderer(model, height=h, width=w)
    rend.update_scene(data, camera=cam)
    img = rend.render()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    try:
        import PIL.Image
        PIL.Image.fromarray(img).save(a.out)
    except Exception:
        import imageio.v2 as imageio
        imageio.imsave(a.out, img)
    print(f"[frame] {a.out} ({w}x{h}) az={a.cam_az} el={a.cam_el} "
          f"dist={a.cam_dist} lookat={a.cam_lookat}")


if __name__ == "__main__":
    main()
