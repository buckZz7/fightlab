"""G1 default scene (scene_29dof.xml) + second robot.

Loads the G1 scene's environment (navy checkerboard, lights, floor,
visual) from scene_29dof.xml, and the robot from g1_29dof.xml.
Duplicates the robot with prefixed names. No design choices.
"""
import os, re
import mujoco

DT = 0.002
G1_SCENE = os.environ.get(
    "G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")


def _extract_balanced(xml, tag):
    """Extract <tag>...</tag> handling nested tags of the same name."""
    start = xml.find(f"<{tag}")
    if start < 0:
        return ""
    depth = 0
    i = start
    while i < len(xml):
        if xml[i:].startswith(f"<{tag}"):
            depth += 1
            i += len(f"<{tag}")
        elif xml[i:].startswith(f"</{tag}>"):
            depth -= 1
            if depth == 0:
                end = i + len(f"</{tag}>")
                inner = xml[start:end]
                # return just the inner content
                open_end = inner.find(">") + 1
                close_start = inner.rfind(f"</{tag}>")
                return inner[open_end:close_start]
            i += len(f"</{tag}>")
        else:
            i += 1
    return ""


def _prefix(xml, pfx):
    xml = re.sub(r'(joint|body|geom|site|actuator|motor|frame|camera|tendon|mesh)'
                 r' name="([^"]+)"',
                 lambda m: f'{m.group(1)} name="{pfx}{m.group(2)}"', xml)
    xml = re.sub(r'joint="([^"]+)"', lambda m: f'joint="{pfx}{m.group(1)}"', xml)
    xml = re.sub(r'body="([^"]+)"', lambda m: f'body="{pfx}{m.group(1)}"', xml)
    xml = re.sub(r'mesh="([^"]+)"', lambda m: f'mesh="{pfx}{m.group(1)}"', xml)
    return xml


def build_default_2bot():
    scene_dir = os.path.dirname(G1_SCENE)
    mesh_dir = os.path.join(scene_dir, "meshes")

    # Read scene (environment: visual, checkerboard, lights, floor)
    with open(G1_SCENE) as f:
        scene_xml = f.read()
    scene_visual = _extract_balanced(scene_xml, "visual")
    scene_asset = _extract_balanced(scene_xml, "asset")
    scene_world = _extract_balanced(scene_xml, "worldbody")

    # Read robot (g1_29dof.xml)
    g1_path = os.path.join(scene_dir, "g1_29dof.xml")
    with open(g1_path) as f:
        g1_xml = f.read()
    g1_inner = re.search(r'<mujoco[^>]*>(.*)</mujoco>', g1_xml, re.DOTALL).group(1)

    robot_default = _extract_balanced(g1_inner, "default")
    robot_asset = _extract_balanced(g1_inner, "asset")
    robot_actu = _extract_balanced(g1_inner, "actuator")
    robot_world = _extract_balanced(g1_inner, "worldbody")

    # r2 = prefixed copy, offset on X
    r2_world = _prefix(robot_world, "r2_").replace('pos="0 0 0.793"', 'pos="0.3 0 0.793"')
    r2_asset = _prefix(robot_asset, "r2_")
    r2_actu = _prefix(robot_actu, "r2_")

    # r1 = prefixed copy, offset on X
    r1_world = _prefix(robot_world, "r1_").replace('pos="0 0 0.793"', 'pos="-0.6 0 0.793"')
    r1_asset = _prefix(robot_asset, "r1_")
    r1_actu = _prefix(robot_actu, "r1_")

    combined = f"""<mujoco model="g1_default_2bot">
  <compiler angle="radian" meshdir="{mesh_dir}" autolimits="true"/>
  <option integrator="RK4" timestep="{DT}" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720" azimuth="-130" elevation="-20"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <default>
    {robot_default}
  </default>

  <asset>
    {scene_asset}
    {r1_asset}
    {r2_asset}
  </asset>

  <worldbody>
    {scene_world}

    <!-- Broadcast tracking camera: follows both robots' center of mass.
         Robots are on the X axis, so camera sits on -Y looking toward +Y.
         Three-quarter front angle: offset on -Y and +X, slightly elevated. -->
    <camera name="broadcast" mode="trackcom"
            pos="-0.15 -4.0 1.2"
            xyaxes="1 0 0 0 0 1"
            fovy="45"/>

    {r1_world}
    {r2_world}
  </worldbody>

  <actuator>
    {r1_actu}
    {r2_actu}
  </actuator>
</mujoco>"""

    model = mujoco.MjModel.from_xml_string(combined)
    model.opt.timestep = DT
    model.opt.tolerance = 1e-4
    model.opt.iterations = 50
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

    # Enable collision on target bodies (torso, head, shoulders, elbows)
    # + weapon bodies (wrists, ankles) + foot bodies. The G1's mesh geoms
    # have collision disabled by default — we re-enable it for combat.
    COMBAT_BODIES = set()
    for pfx in ("r1_", "r2_"):
        for nm in ["torso_link", "head_link",
                   "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                   "left_elbow_link", "right_elbow_link",
                   "left_wrist_yaw_link", "right_wrist_yaw_link",
                   "left_ankle_pitch_link", "left_ankle_roll_link",
                   "right_ankle_pitch_link", "right_ankle_roll_link",
                   "left_knee_link", "right_knee_link"]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{nm}")
            if bid >= 0:
                COMBAT_BODIES.add(bid)

    for i in range(model.ngeom):
        bid = model.geom_bodyid[i]
        if bid in COMBAT_BODIES:
            model.geom_contype[i] = 1
            model.geom_conaffinity[i] = 1
            # For wrist geoms: increase collision margin so punches register
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if "wrist_yaw_link" in bname:
                model.geom_solref[i] = [0.01, 1.0]
                model.geom_solimp[i] = [0.5, 0.9, 0.001, 0.5, 2.0]
                # Use a larger margin for collision detection
                model.geom_margin[i] = 0.02  # 2cm contact margin

    # Firm contacts for foot planting
    model.opt.impratio = 20.0
    for i in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]) or ""
        is_floor = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        is_foot = bname.endswith(("ankle_pitch_link", "ankle_roll_link"))
        if is_floor or is_foot:
            model.geom_solref[i] = [0.008, 1.0]
            model.geom_solimp[i] = [0.9, 0.95, 0.002, 0.5, 2.0]
            model.geom_condim[i] = 4

    return model