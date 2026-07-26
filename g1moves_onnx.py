"""Run a G1 Moves ONNX policy in OUR MuJoCo sim (raw PD controller).

This is the Track B mechanism gate: prove the pretrained full-body prior
runs stably in scene_29dof.xml (NOT through G1SelfPlayEnv, which overwrites
ctrl with its own frozen-base controller).

Exact recipe from experientialtech/g1-moves/run_policy.py:
  obs[160] = ref_jp(29) + ref_jv(29) + anchor_pos_b(3) + anchor_ori_b(6)
             + base_ang_vel(3) + base_lin_vel(3) + (jp-def)(29) + jv(29) + last_act(29)
  action[29] = policy output; target = action + DEFAULT_JOINT_POS
  tau = KP*(target - jp) - KD*jv ; ctrl[:29] = tau
  sim dt = 0.02/4 (200Hz), control every 4 steps.

Usage:
  python g1moves_onnx.py --onnx g1moves/policy/M_Move18_frontkick.onnx \
      --npz g1moves/motion/M_Move18_frontkick.npz \
      --scene /path/to/scene_29dof.xml --out kick.mp4 --fps 50
"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco

# ---- G1 Moves PD gains (29 DOF), matched to training ----
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
KP = np.array([40.2,99.1,40.2,99.1,28.6,28.6, 40.2,99.1,40.2,99.1,28.6,28.6,
               40.2,28.6,28.6, 14.3,14.3,14.3,14.3,14.3,16.8,16.8,
               14.3,14.3,14.3,14.3,14.3,16.8,16.8], dtype=np.float32)
KD = np.array([2.6,6.3,2.6,6.3,1.8,1.8, 2.6,6.3,2.6,6.3,1.8,1.8,
               2.6,1.8,1.8, 0.9,0.9,0.9,0.9,0.9,1.1,1.1,
               0.9,0.9,0.9,0.9,0.9,1.1,1.1], dtype=np.float32)
DECIMATION = 4
CONTROL_DT = 0.02

def quat_to_rot(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def rot_to_6d(R):
    return R[:, :2].T.flatten().astype(np.float32)

def body_frame(robot_pos, robot_quat_wxyz, anchor_pos_w, anchor_quat_wxyz):
    R = quat_to_rot(robot_quat_wxyz)
    dpos = anchor_pos_w - robot_pos
    anchor_pos_b = R.T @ dpos
    R_anchor = quat_to_rot(anchor_quat_wxyz)
    R_rel = R.T @ R_anchor
    anchor_ori_b = rot_to_6d(R_rel)
    return anchor_pos_b.astype(np.float32), anchor_ori_b

def run(onnx_path, npz_path, scene_path, out=None, fps=50, speed=1.0, render=True):
    motion = np.load(npz_path)
    ref_jp = motion["joint_pos"].astype(np.float32)   # (T,29)
    ref_jv = motion["joint_vel"].astype(np.float32)      # (T,29)
    ref_bp = motion["body_pos_w"].astype(np.float64)     # (T,N,3)
    ref_bq = motion["body_quat_w"].astype(np.float64)    # (T,N,4) wxyz
    fps_m = float(motion["fps"]); T = ref_jp.shape[0]

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    model.opt.timestep = CONTROL_DT / DECIMATION

    # find IMU sensor offsets (gyro/acc) in sensordata
    def sadr(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return model.sensor_adr[sid], model.sensor_dim[sid]
    try:
        g0, gd = sadr("imu_gyro"); a0, ad = sadr("imu_acc")
    except Exception:
        g0 = a0 = 0; gd = ad = 3

    last_action = np.zeros(29, dtype=np.float32)
    mtime = 0.0
    nframes = int(T / fps_m * fps) if render else 0

    renderer = None
    if render:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 3.2; cam.elevation = -8; cam.lookat[:] = [0,0,0.8]
        frames = []

    # init pose from frame 0
    data.qpos[:3] = ref_bp[0,0]
    data.qpos[3:7] = ref_bq[0,0]
    data.qpos[7:36] = ref_jp[0]
    mujoco.mj_forward(model, data)

    min_pelvis, fell = 1.0, False
    step = 0
    while mtime < T / fps_m:
        frame = int(mtime * fps_m) % T
        robot_pos = data.qpos[:3].copy()
        robot_quat = data.qpos[3:7].copy()
        jp = data.qpos[7:36].astype(np.float32)
        jv = data.qvel[6:35].astype(np.float32)
        ang = data.sensordata[g0:g0+gd].astype(np.float32) if data.sensordata.size >= g0+gd else np.zeros(3,np.float32)
        lin = data.sensordata[a0:a0+ad].astype(np.float32) if data.sensordata.size >= a0+ad else np.zeros(3,np.float32)
        ap = ref_bp[frame,0].astype(np.float64)
        aq = ref_bq[frame,0].astype(np.float64)
        apb, aob = body_frame(robot_pos, robot_quat, ap, aq)
        obs = np.concatenate([ref_jp[frame], ref_jv[frame], apb, aob,
                             ang, lin, jp - DEFAULT_JOINT_POS, jv, last_action]).astype(np.float32)
        actions = sess.run(["actions"], {"obs": obs[None]})[0][0]
        last_action = actions.copy()
        target = actions + DEFAULT_JOINT_POS
        tau = KP*(target - jp) - KD*jv
        tau = np.clip(tau, -120, 120)  # guard against NaN/explosion
        if not np.all(np.isfinite(tau)):
            tau = np.zeros(29, dtype=np.float32)
        data.ctrl[:29] = tau
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        mtime += CONTROL_DT * speed
        step += 1
        pz = float(data.qpos[2])
        min_pelvis = min(min_pelvis, pz)
        if pz < 0.4 and not fell:
            fell = True
        if render and step % 2 == 0:
            cam.azimuth = 90 + 25*np.sin(step*0.004)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    if render and out:
        imageio.mimsave(out, frames, fps=fps)
        print(f"RENDERED {len(frames)} frames -> {out}")
    print(f"DONE: min_pelvis_z={min_pelvis:.3f} fell={fell} steps={step} "
          f"duration={mtime:.1f}s")
    return {"min_pelvis": min_pelvis, "fell": fell}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    run(a.onnx, a.npz, a.scene, out=a.out, fps=a.fps,
        speed=a.speed, render=not a.no_render)
