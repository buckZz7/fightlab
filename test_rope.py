"""Standalone test: does a compliant rope geom catch a box (no leak, no bounce)?"""
import numpy as np
import mujoco

ROPE_SOLREF = "0.06 1"
ROPE_SOLIMP = "0.9 0.95 0.001"

XML = f"""<mujoco model="ropetest">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="1"/>
    <!-- north rope at y=1.2, compliant -->
    <geom name="rope_n" type="box" pos="0 1.2 0.76" size="1.2 0.015 0.015"
          contype="1" conaffinity="1" solref="{ROPE_SOLREF}" solimp="{ROPE_SOLIMP}"/>
    <!-- free test box -->
    <body name="box" pos="0 1.0 0.76">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.08" mass="2.0" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
bi = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
qoff = 0  # the single free joint lives at qpos[0:7]
d.qpos[qoff + 2] = 0.76
d.qvel[qoff + 1] = 3.0   # shove +y at 3 m/s toward rope at y=1.2
mujoco.mj_forward(m, d)

maxy = 0.0
passed = False
settle = []
for i in range(400):
    mujoco.mj_step(m, d)
    y = float(d.qpos[qoff + 1])
    maxy = max(maxy, y)
    if i > 60:
        settle.append(y)
    if y > 1.3:
        passed = True

print(f"box y max = {maxy:.2f}  (rope at y=1.2)")
print(f"LEAKED past rope : {passed}")
print(f"settled y       = {min(settle):.2f}  (compliant catch, no rebound overshoot)")
