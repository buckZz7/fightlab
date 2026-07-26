"""Build NPZ motion files from G1 Moves PKL/NPZ for use in our sim.

The PKLs expose dof_pos(T,29), root_pos(T,3), root_rot(T,4), fps.
The front-kick NPZ also has body_pos_w/body_quat_w (T,N,3/4).

We synthesize a minimal NPZ with the keys g1moves_onnx.py / the fight env need:
  joint_pos (T,29), joint_vel (T,29) [finite diff], fps,
  body_pos_w (T,1,3) [root pelvis], body_quat_w (T,1,4) [root quat].

For clips that already ship an NPZ (frontkick), we keep it as-is.
"""
import os, sys, pickle, glob
import numpy as np

SRC = "/workspace/repo/g1moves"
OUT = "/workspace/repo/g1moves/motion"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_from_pkl(pkl, npz_out, fps_override=None):
    d = load_pkl(pkl)
    dof = np.asarray(d["dof_pos"], dtype=np.float64)      # (T,29)
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)    # (T,3)
    root_rot = np.asarray(d["root_rot"], dtype=np.float64)   # (T,4) wxyz
    fps = int(d.get("fps", fps_override or 50))
    T = dof.shape[0]
    # joint velocities via finite difference
    jv = np.zeros_like(dof)
    jv[1:] = (dof[1:] - dof[:-1]) * fps
    jv[0] = jv[1]
    # body 0 = pelvis: pos = root_pos, quat = root_rot
    body_pos_w = root_pos[:, None, :].astype(np.float64)      # (T,1,3)
    body_quat_w = root_rot[:, None, :].astype(np.float64)    # (T,1,4)
    np.savez(npz_out,
             joint_pos=dof.astype(np.float32),
             joint_vel=jv.astype(np.float32),
             body_pos_w=body_pos_w.astype(np.float32),
             body_quat_w=body_quat_w.astype(np.float32),
             fps=np.array([fps], dtype=np.float64))
    print(f"built {npz_out}: T={T} fps={fps}")

def main():
    # PKL clips -> NPZ
    pkls = {
        "M_Move2_lowpunch": 50,
        "M_Move7_rapidpunch": 50,
        "M_ShortMove12_quickjab": 50,
        "M_Move10_sidekick": 50,
        "M_ShortMove13_snapkick": 50,
    }
    for name, fps in pkls.items():
        pkl = os.path.join(SRC, "motion", f"{name}.pkl")
        npz = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(pkl) and not os.path.exists(npz):
            build_from_pkl(pkl, npz, fps)
        elif os.path.exists(npz):
            print(f"skip {name} (npz exists)")
        else:
            print(f"MISSING {name}.pkl")
    # frontkick already has NPZ
    print("done")

if __name__ == "__main__":
    main()
