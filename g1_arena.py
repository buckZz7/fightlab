"""G1 two-robot boxing arena builder.

Builds a MuJoCo model with two G1 humanoids facing each other by loading
the G1 spec and duplicating all bodies/joints/actuators with name prefixes.

Uses the compiled-model string approach: load G1 scene XML, prefix all
named references, combine two copies offset on the X axis facing each other.

Ring options (build_arena(ring=...)):
  'ropes' (default) -> regulation SOFT square ring: 4 padded corner
        posts (hard) + 3 rope levels (0.46/0.76/1.07 m) as compliant,
        non-elastic contacts (solref 0.06/1, solimp 0.9/0.95/0.001).
        Bots can lean/corner into ropes and get pushed back, but can't
        walk through. This is the realistic boxing-ring behavior.
  'walls'  -> old hard invisible walls (terminate-on-touch feel).
  'open'   -> no boundary (infinite space).
"""
import re
import mujoco
import numpy as np

import os
G1_SCENE_XML = os.environ.get(
    "G1_SCENE_XML",
    "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")

# Joint indices (29-DoF actuator order: legs 0-11, waist 12-14, arm 15-28)
SKILL_JOINTS = list(range(15, 29))  # arms only — fight policy controls these
N_SKILL = 14
N_QPOS = 36   # 7 freejoint + 29 joints
N_QVEL = 35   # 6 freejoint + 29 joints
DT = 0.002
FRAME_SKIP = 10  # 50 Hz

RESIDUAL_SCALE = np.array(
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2] +   # left arm
    [0.6, 0.4, 0.6, 0.8, 0.2, 0.2, 0.2])    # right arm


def _prefix_xml(xml, prefix):
    """Prefix all name= and mesh= references in a MuJoCo XML string."""
    names = set(re.findall(r'name="([^"]+)"', xml))
    mesh_refs = set(re.findall(r'mesh="([^"]+)"', xml))
    mat_refs = set(re.findall(r'material="([^"]+)"', xml))
    all_refs = names | mesh_refs | mat_refs

    all_refs.discard("floor")

    for ref in sorted(all_refs, key=len, reverse=True):
        xml = xml.replace(f'name="{ref}"', f'name="{prefix}{ref}"')
        xml = xml.replace(f'mesh="{ref}"', f'mesh="{prefix}{ref}"')
        xml = xml.replace(f'material="{ref}"', f'material="{prefix}{ref}"')

    joint_refs = set(re.findall(r'joint="([^"]+)"', xml))
    joint_refs.discard("")
    for ref in sorted(joint_refs, key=len, reverse=True):
        xml = xml.replace(f'joint="{ref}"', f'joint="{prefix}{ref}"')

    xml = re.sub(r'<geom[^>]*name="(?:r1_|r2_)?floor"[^>]*/>', '', xml)
    return xml


# Regulation rope heights (m) above mat: 18/30/42/54 inch (USA Boxing + AIBA).
ROPE_HEIGHTS = (0.46, 0.76, 1.07, 1.37)
ROPE_DIA = 0.025            # 25 mm diameter (>=1in reg)
ROPE_SOLREF = "0.06 1"          # tc=0.06 (compliant), ratio=1 (no bounce)
ROPE_SOLIMP = "0.9 0.95 0.001"  # smooth ramp, near-zero margin
POST_H = 1.47              # 58in above canvas (reg)
POST_R = 0.05               # <=4in diameter (reg)


def _ring_geoms(ring, half):
    if ring == "open":
        return ""
    if ring == "walls":
        return """
    <geom name="wall_n" type="box" pos="0 2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_s" type="box" pos="0 -2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_e" type="box" pos="2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_w" type="box" pos="-2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>"""
    # 'ropes' (default): 4 padded corner posts (hard) + 4 rope levels (soft)
    h = half
    g = []
    # Corner posts: solid, padded look, hard contact. West=red (r1), East=blue (r2)
    post_specs = [(-1, -1, "0.9 0.1 0.1"), (-1, 1, "0.9 0.1 0.1"),
                  (1, -1, "0.1 0.3 0.9"), (1, 1, "0.1 0.3 0.9")]
    for sx, sy, col in post_specs:
        g.append(
            f'<geom name="post_{sx}_{sy}" '
            f'type="cylinder" pos="{sx*h} {sy*h} {POST_H/2}" '
            f'size="{POST_R} {POST_H/2} 0" rgba="{col} 0.8" '
            f'contype="1" conaffinity="1"/>')
    # Ropes: thin compliant bars along each edge at each height
    for lvl in ROPE_HEIGHTS:
        # North/South (along X) at y=+/-h
        g.append(
            f'<geom name="rope_n_{lvl}" type="box" pos="0 {h} {lvl}" '
            f'size="{h} {ROPE_DIA/2} {ROPE_DIA/2}" rgba="0.9 0.1 0.1 0.7" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
        g.append(
            f'<geom name="rope_s_{lvl}" type="box" pos="0 {-h} {lvl}" '
            f'size="{h} {ROPE_DIA/2} {ROPE_DIA/2}" rgba="0.9 0.1 0.1 0.7" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
        # East/West (along Y) at x=+/-h
        g.append(
            f'<geom name="rope_e_{lvl}" type="box" pos="{h} 0 {lvl}" '
            f'size="{ROPE_DIA/2} {h} {ROPE_DIA/2}" rgba="0.9 0.1 0.1 0.7" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
        g.append(
            f'<geom name="rope_w_{lvl}" type="box" pos="{-h} 0 {lvl}" '
            f'size="{ROPE_DIA/2} {h} {ROPE_DIA/2}" rgba="0.9 0.1 0.1 0.7" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
    # Hard invisible backstop just outside ropes (anti-tunnel at speed).
    # Ropes give the compliant catch + visual; backstop guarantees
    # containment so a fast bot can't pass through the thin rope.
    bs = 0.07
    g.append(
        f'<geom name="stop_n" type="box" pos="0 {h+bs} 0.9" size="{h} 0.02 0.9" '
        f'rgba="0 0 0 0" contype="1" conaffinity="1"/>')
    g.append(
        f'<geom name="stop_s" type="box" pos="0 {-h-bs} 0.9" size="{h} 0.02 0.9" '
        f'rgba="0 0 0 0" contype="1" conaffinity="1"/>')
    g.append(
        f'<geom name="stop_e" type="box" pos="{h+bs} 0 0.9" size="0.02 {h} 0.9" '
        f'rgba="0 0 0 0" contype="1" conaffinity="1"/>')
    g.append(
        f'<geom name="stop_w" type="box" pos="{-h-bs} 0 0.9" size="0.02 {h} 0.9" '
        f'rgba="0 0 0 0" contype="1" conaffinity="1"/>')
    return "\n    ".join(g)


def build_arena(ring="ropes", half=2.4):
    """Build a two-G1 boxing arena model.

    ring: 'ropes' (default, soft square ring) | 'walls' | 'open'
    half: ring half-extent (m). Regulation 16-20ft => half 2.4-3.05.
          Default 2.4 (4.8m, min pro) for fight density; tunable
          knob - widening is a cheap warm-start fine-tune, ropes transfer.
    """
    MESH_DIR = os.environ.get(
        "G1_MESH_DIR",
        "/opt/data/unitree_mujoco/unitree_robots/g1/meshes")
    # Load G1 scene XML and add fist collision geoms
    spec = mujoco.MjSpec.from_file(G1_SCENE_XML)
    for side in ("left", "right"):
        wrist = next(b for b in spec.bodies
                     if b.name == f"{side}_wrist_yaw_link")
        wrist.add_geom(
            name=f"{side}_fist_col", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.06], pos=[0.05, 0, 0], mass=0.3,
            rgba=[1, 0, 0, 0.5], contype=1, conaffinity=1)
    xml = spec.to_xml()

    xml_r1 = _prefix_xml(xml, "r1_")
    xml_r2 = _prefix_xml(xml, "r2_")

    def extract_sections(x):
        asset = re.search(r'<asset>(.*?)</asset>', x, re.DOTALL)
        worldbody = re.search(r'<worldbody>(.*?)</worldbody>', x, re.DOTALL)
        actuator = re.search(r'<actuator>(.*?)</actuator>', x, re.DOTALL)
        default = re.search(r'<default>(.*?)</default>', x, re.DOTALL)
        return {
            'asset': asset.group(1) if asset else '',
            'worldbody': worldbody.group(1) if worldbody else '',
            'actuator': actuator.group(1) if actuator else '',
            'default': default.group(1) if default else '',
        }
    r1 = extract_sections(xml_r1)
    r2 = extract_sections(xml_r2)

    r1_body = r1['worldbody'].replace(
        'pos="0 0 0.793"',
        'pos="-0.6 0 0.793"'
    )
    r2_body = r2['worldbody'].replace(
        'pos="0 0 0.793"',
        'pos="0.3 0 0.793"'
    )

    ring_geoms = _ring_geoms(ring, half)

    combined = f"""<mujoco model="g1_boxing_arena">
  <compiler angle="radian" meshdir="{MESH_DIR}" autolimits="true"/>
  <option integrator="RK4" timestep="{DT}" gravity="0 0 -9.81"/>

  <default>
    <joint damping="0.5" armature="0.1"/>
    <geom condim="3" friction="1.0 0.5 0.5"/>
    <default class="torso_motor">
      <joint armature="0.01" damping="0.05" frictionloss="0.2"/>
    </default>
    <default class="leg_motor">
      <joint armature="0.01" damping="0.05" frictionloss="0.2"/>
    </default>
    <default class="ankle_motor">
      <joint armature="0.01" damping="0.05" frictionloss="0.2"/>
    </default>
    <default class="arm_motor">
      <joint armature="0.01" damping="0.05" frictionloss="0.2"/>
    </default>
    <default class="wrist_motor">
      <joint armature="0.01" damping="0.05" frictionloss="0.2"/>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    {r1['asset']}
    {r2['asset']}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    {ring_geoms}

    <!-- Robot 1 (red) -->
    {r1_body}

    <!-- Robot 2 (blue) -->
    {r2_body}
  </worldbody>

  <actuator>
    {r1['actuator']}
    {r2['actuator']}
  </actuator>
</mujoco>"""

    model = mujoco.MjModel.from_xml_string(combined)
    model.opt.timestep = DT
    # Stability for RL random exploration: the elliptic friction cone can
    # raise a rank-deficient sparse-Hessian FatalError on degenerate
    # contacts (random exploration hits these constantly). Looser
    # solver tolerance avoids the singular Hessian without the
    # pyramidal-cone enum (which is a no-op in this mujoco build).
    model.opt.tolerance = 1e-4
    model.opt.iterations = 50
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

    # Performance: disable mesh-mesh self-collision we don't need, BUT keep
    # foot/ankle contact ENABLED -- the G1's feet are mesh geoms only, so
    # disabling them removes foot-ground contact and the robot cannot stand.
    # (Found 2026-07-26: feet had no collision -> PD-to-HOME sagged + fell.)
    TORSO_TARGET_BODIES = {
        "r1_torso_link", "r2_torso_link",
        "r1_left_shoulder_pitch_link", "r1_left_shoulder_roll_link",
        "r1_left_shoulder_yaw_link", "r1_left_elbow_link",
        "r1_right_shoulder_pitch_link", "r1_right_shoulder_roll_link",
        "r1_right_shoulder_yaw_link", "r1_right_elbow_link",
        "r2_left_shoulder_pitch_link", "r2_left_shoulder_roll_link",
        "r2_left_shoulder_yaw_link", "r2_left_elbow_link",
        "r2_right_shoulder_pitch_link", "r2_right_shoulder_roll_link",
        "r2_right_shoulder_yaw_link", "r2_right_elbow_link",
    }
    # Feet = lowest links. Must keep collision for ground contact.
    FOOT_BODIES = {
        "r1_left_ankle_pitch_link", "r1_left_ankle_roll_link",
        "r1_right_ankle_pitch_link", "r1_right_ankle_roll_link",
        "r2_left_ankle_pitch_link", "r2_left_ankle_roll_link",
        "r2_right_ankle_pitch_link", "r2_right_ankle_roll_link",
    }
    KEEP_COLLISION = TORSO_TARGET_BODIES | FOOT_BODIES
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          model.geom_bodyid[i]) or ""
            if body_name not in KEEP_COLLISION:
                model.geom_contype[i] = 0
                model.geom_conaffinity[i] = 0

    return model
