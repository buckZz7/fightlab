"""G1 two-robot boxing arena builder.

Builds a MuJoCo model with two G1 humanoids facing each other by
loading the G1 robot XML, duplicating all bodies/joints/actuators
with name prefixes, and combining two copies offset on the X axis.

Ring options (build_arena(ring=...)):
  'ropes' (default) -> regulation SOFT square ring: 4 padded corner
        posts (hard) + 3 rope levels (0.46/0.76/1.07 m) as compliant,
        non-elastic contacts (solref 0.06/1, solimp 0.9/0.95/0.001).
        Bots can lean/corner into ropes and get pushed back, but can't
        walk through. This is the realistic boxing-ring behavior.
  'walls' -> old hard invisible walls (terminate-on-touch feel).
  'open'   -> no boundary (infinite space).

NOTE on MuJoCo version: this builds the final XML STRING directly
(string manipulation on g1_29dof.xml) and calls
MjModel.from_xml_string(). It does NOT use MjSpec/MjModel.to_xml()
(regex section extraction), which is broken in MuJoCo 3.2.x.
"""
import re
import os
import numpy as np
import mujoco

G1_SCENE_XML = os.environ.get(
    "G1_SCENE_XML",
    "/opt/data/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
# The scene file INCLUDES g1_29dof.xml -- read the real robot directly.
G1_ROBOT_XML = os.path.join(
    os.path.dirname(G1_SCENE_XML), "g1_29dof.xml")
MESH_DIR = os.environ.get(
    "G1_MESH_DIR",
    "/opt/data/unitree_mujoco/unitree_robots/g1/meshes")

# Joint indices (29-DoF actuator order: legs 0-11, waist 12-14, arm 15-28)
SKILL_JOINTS = list(range(15, 29))  # arms only -- fight policy controls these
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
    # drop any floor geom in the robot body (arena adds its own floor)
    xml = re.sub(r'<geom[^>]*name="(?:r1_|r2_)?floor"[^>]*/>', '', xml)
    return xml


def _strip_mujoco_wrapper(xml):
    """Remove the outer <mujoco>...</mujoco> tag, keep inner content.

    The robot file is a full <mujoco model=...> doc; embedding it
    inline would nest <mujoco> inside <worldbody> (schema error).
    """
    m = re.search(r'<mujoco[^>]*>(.*)</mujoco>', xml, re.DOTALL)
    return m.group(1) if m else xml


def _add_fist_geoms(xml):
    """NO fake fist sphere. The G1's wrist_yaw_link ALREADY has
    collision geoms (size 0.06, same as a fist) -- the wrist-end IS
    the bot's hand/fist. We punch with the wrist stub (the G1 has no
    finger bodies to close into a fist). Damage detection uses the
    wrist's own collision geoms, not an added sphere.
    """
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
        return (
            '<geom name="wall_n" type="box" pos="0 2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>'
            '<geom name="wall_s" type="box" pos="0 -2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>'
            '<geom name="wall_e" type="box" pos="2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>'
            '<geom name="wall_w" type="box" pos="-2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>'
        )
    # 'ropes' (default): 4 padded corner posts (hard) + 4 rope levels (soft)
    h = half
    g = []
    # Corner posts: padded cylinders at the 4 corners ONLY. Dark gray
    # (readable against the spotlight, not near-black-invisible) so the
    # ring reads as a silhouette. Slightly inset so they frame, not block.
    inset = 0.12
    post_specs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for sx, sy in post_specs:
        g.append(
            f'<geom name="post_{sx}_{sy}" '
            f'type="cylinder" pos="{sx*(h-inset)} {sy*(h-inset)} {POST_H/2}" '
            f'size="{POST_R} {POST_H/2} 0" rgba="0.30 0.30 0.33 1" '
            f'contype="1" conaffinity="1"/>')
    # Ropes: THICK solid tubes, dark gray vinyl (readable silhouette
    # against the spotlight -- near-black vanished at broadcast angle).
    # Gloves (red/blue) are the color accent.
    rope_rgba = "0.28 0.28 0.31 1"
    for lvl in ROPE_HEIGHTS:
        # North/South (along X) at y=+/-h
        g.append(
            f'<geom name="rope_n_{lvl}" type="box" pos="0 {h} {lvl}" '
            f'size="{h} {ROPE_DIA} {ROPE_DIA}" rgba="{rope_rgba}" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
        g.append(
            f'<geom name="rope_s_{lvl}" type="box" pos="0 {-h} {lvl}" '
            f'size="{h} {ROPE_DIA} {ROPE_DIA}" rgba="{rope_rgba}" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
        # East/West (along Y) at x=+/-h
        g.append(
            f'<geom name="rope_e_{lvl}" type="box" pos="{h} 0 {lvl}" '
            f'size="{ROPE_DIA} {h} {ROPE_DIA}" rgba="{rope_rgba}" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
        g.append(
            f'<geom name="rope_w_{lvl}" type="box" pos="{-h} 0 {lvl}" '
            f'size="{ROPE_DIA} {h} {ROPE_DIA}" rgba="{rope_rgba}" '
            f'contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>')
    # Hard invisible backstop just outside ropes (anti-tunnel at speed).
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
    # Read the real G1 robot XML directly (scene includes it).
    with open(G1_ROBOT_XML) as f:
        robot_xml = f.read()
    # The robot file is a full <mujoco>...</mujoco> doc. Strip the
    # outer wrapper; we want its INNER sections (worldbody/asset/
    # actuator/default), not the <mujoco> tag or nested <worldbody>.
    robot_xml = _strip_mujoco_wrapper(robot_xml)
    # Add fist collision spheres to the wrist bodies.
    robot_xml = _add_fist_geoms(robot_xml)

    # Extract the inner sections we need from the robot XML.
    robot_world = _extract_section(robot_xml, "worldbody")
    robot_asset = _extract_section(robot_xml, "asset")
    robot_actu = _extract_section(robot_xml, "actuator")
    # (robot <default> classes are merged via our own <default> block;
    #  the robot's motor classes are referenced by class= attr in
    #  its bodies/joints, so we re-declare them in our <default>.)

    # Prefix each robot's sections with r1_/r2_.
    r1_world = _prefix_xml(robot_world, "r1_")
    r2_world = _prefix_xml(robot_world, "r2_")
    r1_asset = _prefix_xml(robot_asset, "r1_")
    r2_asset = _prefix_xml(robot_asset, "r2_")
    r1_actu = _prefix_xml(robot_actu, "r1_")
    r2_actu = _prefix_xml(robot_actu, "r2_")

    # Offset the root body of each robot on X (facing each other).
    r1_body = r1_world.replace('pos="0 0 0.793"', 'pos="-0.6 0 0.793"')
    r2_body = r2_world.replace('pos="0 0 0.793"', 'pos="0.3 0 0.793"')

    ring_geoms = _ring_geoms(ring, half)
    # Street fight: NO ring canvas/platform (feet-sink bug + no ring).
    # Just the default checkerboard floor.
    canvas_geom = ""

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
    <headlight diffuse="0.35 0.4 0.5" ambient="0.1 0.12 0.18" specular="0 0 0"/>
    <rgba haze="0.0 0.0 0.0 0"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <!-- DEFAULT MuJoCo checkerboard environment floor (navy grid,
         gradient skybox) -- unchanged, the classic look. -->
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <!-- Ring canvas: a raised LIGHTER platform ON TOP of the default
         checkerboard -- the distinct boxing mat. -->
    <texture type="2d" name="canvas" builtin="flat" rgb1="0.42 0.46 0.55" width="256" height="256"/>
    <material name="canvas" texture="canvas" texuniform="true" reflectance="0.1"/>
    {r1_asset}
    {r2_asset}
  </asset>

  <worldbody>
    <!-- Overhead spotlight on ring, COOL lights so the navy
         checkerboard floor reads navy (not brown). -->
    <light pos="0 0 3.4" dir="0 0 -1" directional="true"
           diffuse="0.9 0.95 1.0" specular="0.4 0.4 0.5" castshadow="true"/>
    <light pos="-1.5 0 3.8" dir="0.35 0 -1" directional="false"
           diffuse="0.45 0.5 0.65" specular="0.25 0.3 0.4"/>
    <light pos="1.5 0 3.8" dir="-0.35 0 -1" directional="false"
           diffuse="0.45 0.5 0.65" specular="0.25 0.3 0.4"/>

    <!-- DEFAULT checkerboard environment floor (unchanged). -->
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    {canvas_geom}

    {ring_geoms}

    <!-- Robot 1 (red) -->
    {r1_body}

    <!-- Robot 2 (blue) -->
    {r2_body}
  </worldbody>

  <actuator>
    {r1_actu}
    {r2_actu}
  </actuator>
</mujoco>"""

    model = mujoco.MjModel.from_xml_string(combined)
    model.opt.timestep = DT
    # --- FIGHTER DISTINCTION: natural G1 bodies, NO gloves (street
    #     fight / real robot combat like Unitree G1 boxing). The
    #     physics fist sphere (fist_col) stays INVISIBLE for damage.
    #     King vs challenger distinction: a small HEADBAND accent
    #     (colored band on the head) -- subtle, not painted bodies.
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  model.geom_bodyid[i]) or ""
        # headband: color the head geom of each robot king=red/chal=blue
        if body.endswith("head_link") or body.endswith("_head"):
            if body.startswith("r1_"):
                model.geom_rgba[i] = [0.85, 0.1, 0.1, 1.0]   # king: RED headband
            elif body.startswith("r2_"):
                model.geom_rgba[i] = [0.15, 0.4, 0.95, 1.0]  # challenger: BLUE headband

    # --- FIGHTER BODY COLOR: REVERTED -- bodies stay the G1's natural
    #     silver/white (accurate to the real robot). Only the gloves
    #     carry the red/blue fighter differentiator.
    # (body-accent coloring removed per design review -- cartoonish.)
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
    # (Found: feet had no collision -> PD-to-HOME sagged + fell.)
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
    # Wrists = the bot's FISTS (RoboStriker: wrist-end is the hand;
    # the G1 has no finger bodies). Keep collision so punches register.
    WRIST_BODIES = {
        "r1_left_wrist_yaw_link", "r1_right_wrist_yaw_link",
        "r2_left_wrist_yaw_link", "r2_right_wrist_yaw_link",
    }
    KEEP_COLLISION = TORSO_TARGET_BODIES | FOOT_BODIES | WRIST_BODIES
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          model.geom_bodyid[i]) or ""
            if body_name not in KEEP_COLLISION:
                model.geom_contype[i] = 0
                model.geom_conaffinity[i] = 0

    # FIRM CONTACTS -- a ~40kg humanoid on soft MuJoCo contacts (solref
    # timeconst 0.02) slowly SINKS, which reads as "PD can't hold the G1"
    # but is actually solver compliance. Stiffen floor + foot geoms so the
    # feet plant firmly (normal spring timeconst 0.008, full damping) and
    # raise impedance ratio so normal/friction are balanced. Also condim=4
    # (pyramidal friction) for foot grip.
    model.opt.impratio = 20.0
    for i in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   model.geom_bodyid[i]) or ""
        is_floor = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        is_foot = bname in FOOT_BODIES
        if is_floor or is_foot:
            model.geom_solref[i] = [0.008, 1.0]
            model.geom_solimp[i] = [0.9, 0.95, 0.002, 0.5, 2.0]
            model.geom_condim[i] = 4

    return model


def _extract_section(xml, tag):
    """Extract the contents of <tag>...</tag> from the robot XML."""
    m = re.search(rf'<{tag}>(.*?)</{tag}>', xml, re.DOTALL)
    return m.group(1) if m else ""

