"""G1 two-robot boxing arena builder.

Builds a MuJoCo model with two G1 humanoids facing each other by loading
the G1 spec and duplicating all bodies/joints/actuators with name prefixes.

Uses the compiled-model string approach: load G1 scene XML, prefix all
named references, combine two copies offset on the y-axis facing each other.
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
    # Collect all unique names from name="..." attributes
    names = set(re.findall(r'name="([^"]+)"', xml))
    # Also collect mesh="..." and material="..." references
    mesh_refs = set(re.findall(r'mesh="([^"]+)"', xml))
    mat_refs = set(re.findall(r'material="([^"]+)"', xml))
    all_refs = names | mesh_refs | mat_refs

    # Don't prefix "floor" — we'll use our own floor, not the robot's
    all_refs.discard("floor")

    # Sort by length (longest first) to avoid partial replacements
    for ref in sorted(all_refs, key=len, reverse=True):
        # Replace name="ref" -> name="prefix+ref"
        xml = xml.replace(f'name="{ref}"', f'name="{prefix}{ref}"')
        # Replace mesh="ref" -> mesh="prefix+ref"
        xml = xml.replace(f'mesh="{ref}"', f'mesh="{prefix}{ref}"')
        # Replace material="ref" -> material="prefix+ref"
        xml = xml.replace(f'material="{ref}"', f'material="{prefix}{ref}"')

    # Fix actuator joint="..." references
    joint_refs = set(re.findall(r'joint="([^"]+)"', xml))
    joint_refs.discard("")  # safety
    for ref in sorted(joint_refs, key=len, reverse=True):
        xml = xml.replace(f'joint="{ref}"', f'joint="{prefix}{ref}"')

    # Remove the floor geom from the robot's worldbody (we have our own)
    xml = re.sub(r'<geom[^>]*name="(?:r1_|r2_)?floor"[^>]*/>', '', xml)

    return xml


def build_arena():
    """Build a two-G1 boxing arena model with fist collision spheres."""
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

    # Create two prefixed copies
    xml_r1 = _prefix_xml(xml, "r1_")
    xml_r2 = _prefix_xml(xml, "r2_")

    # Extract the body tree and assets from each
    # We need: <asset> section, <worldbody> body tree, <actuator> section,
    # <default> section (for joint/geom defaults)

    # Strip the <mujoco> wrapper
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

    # Build combined XML
    # Robot 1 faces +Y, robot 2 faces -Y, separated on X axis
    # Offset pelvis positions: r1 at x=-0.6, r2 at x=0.6
    # Rotate r2 180 degrees around Z so they face each other
    r1_body = r1['worldbody'].replace(
        'pos="0 0 0.793"',
        'pos="-0.6 0 0.793"'
    )
    r2_body = r2['worldbody'].replace(
        'pos="0 0 0.793"',
        'pos="0.3 0 0.793"'  # close enough for punches to reach
    )

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

    <!-- Arena boundary walls (invisible collision) -->
    <geom name="wall_n" type="box" pos="0 2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_s" type="box" pos="0 -2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_e" type="box" pos="2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>
    <geom name="wall_w" type="box" pos="-2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.1" contype="0" conaffinity="0"/>

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

    # Performance: disable collision on mesh geoms EXCEPT those on torso
    # subtree bodies (which are valid punch targets). This keeps fist-to-
    # torso contact working while removing the expensive mesh-mesh collision
    # on legs/hips that we don't need.
    # Fist spheres (type=SPHERE) and ankle spheres (type=SPHERE) keep collision.
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
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          model.geom_bodyid[i]) or ""
            if body_name not in TORSO_TARGET_BODIES:
                model.geom_contype[i] = 0
                model.geom_conaffinity[i] = 0

    return model
