"""Test: compliant rope + hard invisible backstop behind it (anti-tunnel)."""
import numpy as np
import mujoco

HALF = 2.4
ROPE_Y = 1.37
XML = f"""<mujoco model="rt">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="1"/>
    <!-- compliant visible rope -->
    <geom name="rope_n" type="box" pos="0 {HALF} {ROPE_Y}"
          size="{HALF} 0.0125 0.0125" contype="1" conaffinity="1"
          solref="0.06 1" solimp="0.9 0.95 0.001"/>
    <!-- hard invisible backstop just outside ropes -->
    <geom name="stop_n" type="box" pos="0 {HALF+0.06} {ROPE_Y}"
          size="{HALF} 0.02 0.06" contype="1" conaffinity="1"/>
    <body name="box" pos="0 {HALF-0.2} {ROPE_Y}">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.08" mass="2.0" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
qoff = 0
d.qpos[qoff + 2] = ROPE_Y
d.qvel[qoff + 1] = 4.0   # faster: 4 m/s into rope
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
    if y > HALF + 0.15:
        passed = True

print(f"max y = {maxy:.2f}  (backstop at {HALF+0.06:.2f})")
print(f"LEAKED past backstop : {passed}")
print(f"settled y           = {min(settle):.2f}  (caught, no tunnel)")
