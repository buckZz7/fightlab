import g1_arena as A
robot_xml = open(A.G1_ROBOT_XML).read()
robot_xml = A._add_fist_geoms(robot_xml)
asset_sec = A._extract_section(robot_xml, "asset")
actu_sec  = A._extract_section(robot_xml, "actuator")
r1_asset = A._prefix_xml(asset_sec, "r1_")
r2_asset = A._prefix_xml(asset_sec, "r2_")
r1_actu  = A._prefix_xml(actu_sec,  "r1_")
r2_actu  = A._prefix_xml(actu_sec, "r2_")
xml_r1 = A._prefix_xml(robot_xml, "r1_")
xml_r2 = A._prefix_xml(robot_xml, "r2_")
r1_body = xml_r1.replace('pos="0 0 0.793"', 'pos="-0.6 0 0.793"')
r2_body = xml_r2.replace('pos="0 0 0.793"', 'pos="0.3 0 0.793"')
rg = A._ring_geoms("ropes", 2.4)

combined = f"""<mujoco model="g1_boxing_arena">
  <compiler angle="radian" meshdir="{A.MESH_DIR}" autolimits="true"/>
  <option integrator="RK4" timestep="{A.DT}" gravity="0 0 -9.81"/>

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
    {r1_asset}
    {r2_asset}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    {rg}

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

open("/tmp/combined.xml", "w").write(combined)
# Show line 144 area
lines = combined.splitlines()
for i in range(138, min(152, len(lines))):
    print(f"{i+1}: {lines[i][:100]}")
