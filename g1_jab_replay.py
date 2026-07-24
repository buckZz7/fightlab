"""Scripted jab replay over the frozen LocoBase29 balance base.

Milestone per the architecture research (option 1): validate the interface
before any RL. Hand-scripted jab trajectory for the right arm:
guard -> extend (jab) -> retract -> guard, ~0.5s strike phase.

The punch env adds a heavy bag in front; a real jab should swing the bag.
Pass = robot stays standing through 20 jabs and the bag registers hits.

Right arm joints (29-DoF order, slice 22:29):
  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch, wrist_roll, wrist_yaw
"""
import mujoco
import numpy as np
import sys
sys.path.insert(0, "/opt/data/fightlab-repo-new")

from loco_base29 import LocoBase29, HOME

XML = "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

# Jab trajectory: keyframes for the right arm (offsets from HOME, radians)
# Order: r_sh_pitch, r_sh_roll, r_sh_yaw, r_elbow, r_wr_pitch, r_wr_roll, r_wr_yaw
# NOTE: G1 forward reach maxes ~0.3m at z~1.1-1.2; shoulder_pitch NEGATIVE = forward.
GUARD = np.array([-0.5, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0])     # fist up at chin, elbow bent
EXTEND = np.array([-1.5, 0.12, 0.0, 0.15, 0.0, 0.0, 0.0])  # arm straight forward, slight lift
R_ARM = slice(22, 29)

# phase durations in seconds
T_GUARD_UP = 1.0
T_JAB_OUT = 0.12     # fast extension — punch speed comes from here
T_HOLD = 0.05
T_JAB_BACK = 0.35
T_REST = 0.8


def jab_offset(t):
    """Offset for right arm at time t within one jab cycle."""
    cycle = T_GUARD_UP + T_JAB_OUT + T_HOLD + T_JAB_BACK + T_REST
    t = t % cycle
    if t < T_GUARD_UP:
        # ease into guard
        s = t / T_GUARD_UP
        return GUARD * (s * s * (3 - 2 * s))
    t -= T_GUARD_UP
    if t < T_JAB_OUT:
        s = t / T_JAB_OUT
        return GUARD + (EXTEND - GUARD) * (s * s * (3 - 2 * s))
    t -= T_JAB_OUT
    if t < T_HOLD:
        return EXTEND
    t -= T_HOLD
    if t < T_JAB_BACK:
        s = t / T_JAB_BACK
        return EXTEND + (GUARD - EXTEND) * (s * s * (3 - 2 * s))
    return GUARD


def run(with_bag=True, seconds=15.0, render=None, waist_drive=True):
    m = mujoco.MjModel.from_xml_path(XML)
    if with_bag:
        spec = mujoco.MjSpec.from_file(XML)
        # G1 wrists ship contype=0 (no collision) — add fist collision spheres
        for side in ("left", "right"):
            w = next(b for b in spec.bodies if b.name == f"{side}_wrist_yaw_link")
            w.add_geom(name=f"{side}_fist_col", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[0.06], pos=[0.05, 0, 0], mass=0.3,
                       rgba=[1, 0, 0, 0.5], contype=1, conaffinity=1)
        world = spec.worldbody
        stand = world.add_body(name="bag_stand", pos=[0.30, -0.12, 0.0])
        stand.add_geom(name="stand_pole", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[0.03, 0.5, 0], pos=[0, 0, 0.5],
                       rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
        bag = stand.add_body(name="heavy_bag", pos=[0, 0, 1.0])
        bag.add_joint(name="bag_swing", type=mujoco.mjtJoint.mjJNT_SLIDE,
                      axis=[1, 0, 0], range=[-0.6, 0.6], damping=8.0,
                      stiffness=40.0)
        bag.add_geom(name="bag", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                     size=[0.12], mass=20.0, rgba=[0.8, 0.2, 0.2, 1])
        m = spec.compile()
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002

    d.qpos[2] = 0.75
    d.qpos[7:36] = HOME
    mujoco.mj_forward(m, d)

    loco = LocoBase29()
    pelvis = m.body("pelvis").id
    lo = m.actuator_ctrlrange[:, 0]
    hi = m.actuator_ctrlrange[:, 1]

    bag_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bag_swing") if with_bag else -1
    bag_dof = int(m.jnt_dofadr[bag_jnt]) if bag_jnt >= 0 else None

    steps = int(seconds / 0.002)
    min_z = 1.0
    max_bag_vel = 0.0
    hits = 0
    frames = []
    renderer = mujoco.Renderer(m, 480, 640) if render else None

    for i in range(steps):
        t = i * 0.002
        off = jab_offset(t)
        loco.update(d.qpos, d.qvel)
        target = loco.target.copy()
        target[R_ARM] += off
        if waist_drive:
            # rotate into the punch: waist yaw winds back during guard,
            # snaps through during extension (right jab = rotate left-to-right)
            cyc_t = t % (T_GUARD_UP + T_JAB_OUT + T_HOLD + T_JAB_BACK + T_REST)
            if cyc_t < T_GUARD_UP:
                waist_yaw = -0.25 * (cyc_t / T_GUARD_UP)          # wind up
            elif cyc_t < T_GUARD_UP + T_JAB_OUT + T_HOLD:
                waist_yaw = -0.25 + 0.5 * ((cyc_t - T_GUARD_UP) / (T_JAB_OUT + T_HOLD))  # snap
            else:
                waist_yaw = 0.25 - 0.25 * min(1.0, (cyc_t - T_GUARD_UP - T_JAB_OUT - T_HOLD) / T_JAB_BACK)
            target[12:15] += np.array([waist_yaw, 0.0, 0.0])
        tau = loco.pd_torque(d.qpos, d.qvel, target_override=target)
        d.ctrl[:] = np.clip(tau, lo, hi)
        mujoco.mj_step(m, d)

        z = float(d.xpos[pelvis][2])
        min_z = min(min_z, z)
        if bag_dof is not None:
            bv = abs(float(d.qvel[bag_dof]))
            if bv > max_bag_vel:
                max_bag_vel = bv
            if bv > 0.5:
                hits += 1
        if renderer and i % 33 == 0:
            renderer.update_scene(d, camera="track")
            frames.append(renderer.render())
        if z < 0.4:
            break

    stood = z >= 0.4
    print(f"jab replay: stood={'YES' if stood else 'NO'} min_z={min_z:.2f} "
          f"t={i*0.002:.1f}s max_bag_vel={max_bag_vel:.2f} m/s hit_steps={hits}")
    if render and frames:
        import imageio
        imageio.mimsave(render, frames, fps=30)
        print(f"video: {render} ({len(frames)} frames)")
    return stood, max_bag_vel


if __name__ == "__main__":
    import sys as _sys
    render = "jab_replay.mp4" if "--render" in _sys.argv else None
    run(with_bag=True, seconds=15.0, render=render)
