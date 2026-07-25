"""Replay a G1 Moves full-body ONNX policy in our G1 MuJoCo sim.

Validates Track B: downloaded full-body punch/kick policies drive the Unitree
G1 in OUR scene. Obs + PD gains replicated exactly from G1 Moves run_policy.py
(https://github.com/experientialtech/g1-moves). Headless OSMesa rendering.
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import onnxruntime as ort
import imageio.v2 as imageio

XML = os.environ.get("G1_SCENE_XML", "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
DECIMATION = 4
CONTROL_DT = 0.02

KP = np.array([40.2,99.1,40.2,99.1,28.6,28.6, 40.2,99.1,40.2,99.1,28.6,28.6,
               40.2,28.6,28.6, 14.3,14.3,14.3,14.3,14.3,16.8,16.8,
               14.3,14.3,14.3,14.3,14.3,16.8,16.8], dtype=np.float32)
KD = np.array([2.6,6.3,2.6,6.3,1.8,1.8, 2.6,6.3,2.6,6.3,1.8,1.8,
               2.6,1.8,1.8, 0.9,0.9,0.9,0.9,0.9,1.1,1.1,
               0.9,0.9,0.9,0.9,0.9,1.1,1.1], dtype=np.float32)

def quat_to_rot_matrix(q_wxyz):
    w,x,y,z = q_wxyz
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def rot_to_6d(R):
    return R[:, :2].T.flatten()

def transform_to_body(pos_w, quat_wxyz, anchor_pos_w, anchor_quat_wxyz):
    R_robot = quat_to_rot_matrix(quat_wxyz)
    delta = anchor_pos_w - pos_w
    anchor_pos_b = R_robot.T @ delta
    R_anchor = quat_to_rot_matrix(anchor_quat_wxyz)
    R_rel = R_robot.T @ R_anchor
    anchor_ori_b = rot_to_6d(R_rel)
    return anchor_pos_b.astype(np.float32), anchor_ori_b.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--out", default="/tmp/g1moves_replay.mp4")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--max-frames", type=int, default=100000)
    args = ap.parse_args()

    motion = np.load(args.npz)
    ref_jp = motion["joint_pos"].astype(np.float32)
    ref_jv = motion["joint_vel"].astype(np.float32)
    ref_bp = motion["body_pos_w"].astype(np.float64)
    ref_bq = motion["body_quat_w"].astype(np.float64)
    fps = float(motion["fps"])
    N = ref_jp.shape[0]

    sess = ort.InferenceSession(args.policy)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    model.opt.timestep = CONTROL_DT / DECIMATION

    last_action = np.zeros(29, dtype=np.float32)
    motion_time = 0.0
    renderer = mujoco.Renderer(model, height=480, width=640)
    # init
    data.qpos[2] = 0.78
    data.qpos[7:36] = ref_jp[0]
    mujoco.mj_forward(model, data)

    frames = []
    min_z = 1.0
    frame = 0
    steps = min(args.max_frames, N * 3)
    for _ in range(steps):
        frame = int(motion_time * fps) % N
        rjp = ref_jp[frame]; rjv = ref_jv[frame]
        robot_pos = data.qpos[:3].copy()
        robot_quat = data.qpos[3:7].copy()
        joint_pos = data.qpos[7:36].astype(np.float32)
        joint_vel = data.qvel[6:35].astype(np.float32)
        # IMU: gyro at sensordata[91:94], acc at [94:97]
        ang = data.sensordata[91:94].astype(np.float32) if data.sensordata.size >= 97 else np.zeros(3, np.float32)
        # linear velocity in body frame from qvel
        Rb = quat_to_rot_matrix(robot_quat).T
        lin = (Rb @ data.qvel[0:3]).astype(np.float32)
        a_pos_b, a_ori_b = transform_to_body(robot_pos, robot_quat, ref_bp[frame,0], ref_bq[frame,0])
        obs = np.concatenate([rjp, rjv, a_pos_b, a_ori_b, ang, lin,
                              joint_pos - DEFAULT_JOINT_POS, joint_vel, last_action]).astype(np.float32)
        actions = sess.run(["actions"], {"obs": obs[None]})[0][0]
        last_action = actions.copy()
        target = actions + DEFAULT_JOINT_POS
        data.ctrl[:29] = KP * (target - joint_pos) - KD * joint_vel
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        motion_time += CONTROL_DT
        min_z = min(min_z, data.qpos[2])
        renderer.update_scene(data)
        frames.append(renderer.render())
        if data.qpos[2] < 0.4:
            print(f"FELL at step, pelvis_z={data.qpos[2]:.3f}")
            break

    imageio.mimsave(args.out, frames, fps=int(args.fps))
    print(f"saved {args.out} | frames={len(frames)} | min_pelvis_z={min_z:.3f} | stayed_up={min_z>0.5}")

if __name__ == "__main__":
    main()
