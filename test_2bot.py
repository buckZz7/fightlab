"""Test 2-bot scene: verify both robots can stand using walker."""
import sys, os, json, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MUJOCO_GL"] = "egl"

# Load walker code
_env = {"__file__": "run.py", "__name__": "__main__"}
exec(open("run.py").read().split("def main")[0], _env)
G1Controller = _env["G1Controller"]
ONNXPolicy = _env["ONNXPolicy"]

m = mujoco.MjModel.from_xml_path("scene_2bot.xml")
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)

cfg = json.load(open("model_config.json"))

# Find the qpos offset for r2
# r1 root is at qpos[0:7], r2 root is after all r1 joints
# r1 has 1 free joint (7 qpos) + 43 hinge joints (43 qpos) = 50 qpos
# But the scene has 88 joints total. Let's find r2's root.
r1_joints = []
r2_joints = []
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    if name.startswith("r2_"):
        r2_joints.append((i, name, m.jnt_qposadr[i], m.jnt_dofadr[i]))
    elif not name.startswith("r2_") and name:
        r1_joints.append((i, name, m.jnt_qposadr[i], m.jnt_dofadr[i]))

print(f"r1 joints: {len(r1_joints)}, r2 joints: {len(r2_joints)}")
print(f"r1 qpos range: 0 to {r1_joints[-1][2]}")
print(f"r2 qpos range: {r2_joints[0][2]} to {r2_joints[-1][2]}")

# r2 qpos offset
r2_qpos_offset = r2_joints[0][2] - 7  # r2's root starts 7 before its first joint
print(f"r2 root qpos offset: {r2_qpos_offset}")

# r1 pelvis
r1_pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
r2_pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r2_pelvis")
print(f"r1 pelvis body: {r1_pelvis}, r2 pelvis body: {r2_pelvis}")

# Set default poses for both robots
jn = cfg["joint_names"]
dp = np.array([cfg["default_joint_pos"][j] for j in jn])

# r1 defaults
for name in jn:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        d.qpos[m.jnt_qposadr[jid]] = cfg["default_joint_pos"][name]

# r2 defaults
for name in jn:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r2_{name}")
    if jid >= 0:
        d.qpos[m.jnt_qposadr[jid]] = cfg["default_joint_pos"][name]

mujoco.mj_forward(m, d)
print(f"\nStart: r1 z={d.xpos[r1_pelvis][2]:.4f}, r2 z={d.xpos[r2_pelvis][2]:.4f}")

# Create walker sessions
walker1 = ONNXPolicy("walker.onnx")
walker2 = ONNXPolicy("walker.onnx")
croucher = ONNXPolicy("croucher.onnx")
rotator = ONNXPolicy("rotator.onnx")

# r1 controller (default — reads qpos starting at 0)
ctrl1 = G1Controller(m, d, walker1, croucher, rotator, cfg, None)
ctrl1._cache_actuator_ids()

# r2 controller — we need to modify it to use r2_ prefixed names and r2 qpos offset
# The G1Controller hardcodes qpos indices as 7+i. For r2, it's r2_qpos_offset+7+i.
# Let's monkey-patch it.
class R2Controller(G1Controller):
    def __init__(self, model, data, walker, croucher, rotator, config, reacher, qpos_offset):
        self._r2_offset = qpos_offset
        super().__init__(model, data, walker, croucher, rotator, config, reacher)

    def _build_joint_mappings(self):
        super()._build_joint_mappings()
        # Override: shift all qpos/qvel indices by r2 offset
        # qpos offset for r2's joints
        r2_first_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r2_left_hip_pitch_joint")
        r2_qpos_off = self.model.jnt_qposadr[r2_first_jid] - 7  # r2 root is 7 before first joint
        r2_dof_off = self.model.jnt_dofadr[r2_first_jid] - 6    # r2 root has 6 dof
        for name in self.joint_qpos_indices:
            self.joint_qpos_indices[name] += r2_qpos_off
        for name in self.joint_qvel_indices:
            self.joint_qvel_indices[name] += r2_dof_off

    def _get_base_pose(self):
        d = self.data
        q = self._r2_offset
        return d.qpos[q:q+3].copy(), d.qpos[q+3:q+7].copy()

    def _get_base_velocities(self):
        d = self.data
        # qvel offset for r2's root: find the dof address of r2's root free joint
        # r2's first named joint gives us the dof offset
        r2_first_dof = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r2_left_hip_pitch_joint")
        r2_root_dof = self.model.jnt_dofadr[r2_first_dof] - 6  # root has 6 dof
        lin_vel_world = d.qvel[r2_root_dof:r2_root_dof+3].copy()
        ang_vel_body = d.qvel[r2_root_dof+3:r2_root_dof+6].copy()
        _, quat = self._get_base_pose()
        return self._quat_apply_inverse(quat, lin_vel_world), ang_vel_body

    def _cache_actuator_ids(self):
        self.actuator_ids = []
        for name in self.joint_names:
            act_name = f"r2_{name}"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self.actuator_ids.append(aid)

# Calculate r2 offset: r2's root free joint qpos starts where?
# r1 has 1 free joint (qpos[0:7]) + 43 named joints
# r2's free joint should be right after
r2_root_joint = None
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    if name == "r2_pelvis" or (name.startswith("r2_") and m.jnt_type[i] == 0):
        r2_root_joint = i
        break
# Actually the free joint might not have a name. Let's find it differently.
# r2's first named joint is r2_left_hip_pitch_joint
r2_first = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "r2_left_hip_pitch_joint")
r2_first_qpos = m.jnt_qposadr[r2_first]
# r2's root free joint is 7 qpos before the first named joint
r2_root_qpos = r2_first_qpos - 7
print(f"r2 root qpos: {r2_root_qpos}, r2 first joint qpos: {r2_first_qpos}")

ctrl2 = R2Controller(m, d, walker2, croucher, rotator, cfg, None, r2_root_qpos)
ctrl2._cache_actuator_ids()

# Run both walkers for 12000 steps (60 seconds)
for t in range(12000):
    # r1
    targets1 = ctrl1.step()
    for i, aid in enumerate(ctrl1.actuator_ids):
        if aid >= 0:
            d.ctrl[aid] = targets1[i]
    # r2
    targets2 = ctrl2.step()
    for i, aid in enumerate(ctrl2.actuator_ids):
        if aid >= 0:
            d.ctrl[aid] = targets2[i]
    mujoco.mj_step(m, d)
    if t in [999, 2999, 5999, 8999, 11999]:
        time_s = t * 0.005
        print(f"t={t} ({time_s:.0f}s): r1 z={d.xpos[r1_pelvis][2]:.4f}, r2 z={d.xpos[r2_pelvis][2]:.4f}")
