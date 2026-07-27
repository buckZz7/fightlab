"""FightLab Combat Environment v2 — walker-based.

Architecture:
  - Walker (ONNX) maintains balance for BOTH robots
  - Combat policy outputs: [vx, vy, yaw, 14 arm residuals] = 17D
  - Walker produces 29 joint targets per robot
  - Arm joints (indices 15-28) overridden by combat policy
  - Legs + waist stay from walker (balance guaranteed)
  - Damage: wrist-to-torso contact (punches only, no kicks)
  - Scoring: 10-point must, KO/decision

Open interface for miners:
  - Observation format: customizable (default provided)
  - Action space: 17D (3 velocity + 14 arm residuals)
  - Walker: shared (or custom if miner trains their own)
  - Training: up to miner (PPO, SAC, imitation, whatever)
"""
import os, sys, json, math
import numpy as np
import mujoco
import gymnasium as gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Load walker code from run.py
_env = {"__file__": os.path.join(SCRIPT_DIR, "run.py"), "__name__": "__main__"}
with open(os.path.join(SCRIPT_DIR, "run.py")) as _f:
    exec(_f.read().split("def main")[0], _env)
ONNXPolicy = _env["ONNXPolicy"]
G1Controller = _env["G1Controller"]

# Constants
ACT_DIM = 17  # 3 velocity + 14 arm residuals
ARM_INDICES = list(range(15, 29))
MAX_HP = 100.0
KO_HP = 0.0
FALL_Z = 0.4  # pelvis below this = fall
DT = 0.005


class R2Controller(G1Controller):
    """G1Controller for the second robot (r2_ prefixed)."""

    def __init__(self, model, data, walker, croucher, rotator, config, reacher, qpos_offset, dof_offset):
        self._r2_qpos_off = qpos_offset
        self._r2_dof_off = dof_offset
        super().__init__(model, data, walker, croucher, rotator, config, reacher)

    def _build_joint_mappings(self):
        super()._build_joint_mappings()
        r2_qpos_off = self._r2_qpos_off
        r2_dof_off = self._r2_dof_off
        for name in self.joint_qpos_indices:
            self.joint_qpos_indices[name] += r2_qpos_off
        for name in self.joint_qvel_indices:
            self.joint_qvel_indices[name] += r2_dof_off

    def _get_base_pose(self):
        d = self.data
        q = self._r2_qpos_off
        return d.qpos[q:q+3].copy(), d.qpos[q+3:q+7].copy()

    def _get_base_velocities(self):
        d = self.data
        q = self._r2_dof_off
        lin_vel_world = d.qvel[q:q+3].copy()
        ang_vel_body = d.qvel[q+3:q+6].copy()
        _, quat = self._get_base_pose()
        return self._quat_apply_inverse(quat, lin_vel_world), ang_vel_body

    def _cache_actuator_ids(self):
        self.actuator_ids = []
        for name in self.joint_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"r2_{name}")
            self.actuator_ids.append(aid)


class FightEnv(gym.Env):
    """FightLab combat environment.

    Two G1s with walker balance. Combat policy controls r1.
    Opponent (r2) can be: scripted, another PPO model, or sandbag (PD stand).
    """

    def __init__(self, max_steps=6000, opponent=None, randomize=False):
        super().__init__()
        self.max_steps = max_steps
        self.opponent = opponent  # "sandbag", "scripted:jabbler", or PPO model path
        self.randomize = randomize
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]

        # Load scene
        scene_path = os.path.join(SCRIPT_DIR, "scene_2bot.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        # Load config
        self.cfg = json.load(open(os.path.join(SCRIPT_DIR, "model_config.json")))
        self.joint_names = self.cfg["joint_names"]
        self.default_pos = np.array([self.cfg["default_joint_pos"][j] for j in self.joint_names])

        # Create walker controllers
        walker1 = ONNXPolicy(os.path.join(SCRIPT_DIR, "walker.onnx"))
        walker2 = ONNXPolicy(os.path.join(SCRIPT_DIR, "walker.onnx"))
        croucher = ONNXPolicy(os.path.join(SCRIPT_DIR, "croucher.onnx"))
        rotator = ONNXPolicy(os.path.join(SCRIPT_DIR, "rotator.onnx"))

        self.ctrl1 = G1Controller(self.model, self.data, walker1, croucher, rotator, self.cfg, None)
        self.ctrl1._cache_actuator_ids()

        # Find r2 offsets
        r2_first = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "r2_left_hip_pitch_joint")
        r2_qpos_off = self.model.jnt_qposadr[r2_first] - 7
        r2_dof_off = self.model.jnt_dofadr[r2_first] - 6
        self.r2_qpos_off = r2_qpos_off
        self.r2_dof_off = r2_dof_off

        self.ctrl2 = R2Controller(self.model, self.data, walker2, croucher, rotator, self.cfg, None,
                                   r2_qpos_off, r2_dof_off)
        self.ctrl2._cache_actuator_ids()

        # Body IDs
        self.pelvis_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r2_pelvis"),
        ]

        # Damage detection: wrist + torso body IDs
        self._setup_damage_ids()

        # Load opponent policy if needed
        self.opp_policy = None
        if opponent and opponent.endswith(".zip") and os.path.exists(opponent):
            from stable_baselines3 import PPO
            self.opp_policy = PPO.load(opponent)

        # Action/observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(ACT_DIM,), dtype=np.float32)
        # Default obs: [rel_pos(3), rel_vel(3), self_joints(29), self_hp(1), opp_hp(1),
        #               self_facing(1), dist(1), self_pelvis_z(1), opp_pelvis_z(1)] = 70
        self.obs_dim = 70
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                  shape=(self.obs_dim,), dtype=np.float32)

        # Torque noise (for domain randomization)
        self.torque_noise_std = 0.0

    def _setup_damage_ids(self):
        """Find wrist and torso body IDs for damage detection."""
        self.fist_bodies = [[], []]
        self.torso_bodies = [[], []]
        prefixes = ["", "r2_"]

        # Wrist bodies (weapons — punches only)
        wrist_names = ["left_wrist_yaw_link", "right_wrist_yaw_link",
                       "left_wrist_pitch_link", "right_wrist_pitch_link"]
        # Torso target bodies
        torso_names = ["torso_link", "head_link",
                       "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                       "left_shoulder_roll_link", "right_shoulder_roll_link",
                       "left_shoulder_yaw_link", "right_shoulder_yaw_link",
                       "left_elbow_link", "right_elbow_link"]

        for ai, pfx in enumerate(prefixes):
            for name in wrist_names:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{name}")
                if bid >= 0:
                    self.fist_bodies[ai].append(bid)
            for name in torso_names:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{name}")
                if bid >= 0:
                    self.torso_bodies[ai].append(bid)

        # Geom body mapping for contact detection
        self.geom_body = self.model.geom_bodyid.copy()

    def _place(self):
        """Reset both robots to default pose, facing each other."""
        mujoco.mj_resetData(self.model, self.data)
        # r1
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.cfg["default_joint_pos"][name]
        # r2
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"r2_{name}")
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.cfg["default_joint_pos"][name]
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        """Default observation for the combat policy."""
        d = self.data
        p1 = d.xpos[self.pelvis_ids[0]]
        p2 = d.xpos[self.pelvis_ids[1]]
        rel_pos = (p2 - p1).astype(np.float32)
        rel_vel = (d.cvel[self.pelvis_ids[1]][:3] - d.cvel[self.pelvis_ids[0]][:3]).astype(np.float32)

        # Self joint positions (relative to default)
        jp = np.zeros(29, dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                jp[i] = d.qpos[self.model.jnt_qposadr[jid]] - self.default_pos[i]

        dist = float(np.linalg.norm(rel_pos))
        # Facing: dot of r1's forward direction with direction to r2
        R = np.eye(3).flatten()
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        facing = float(np.dot(R.reshape(3,3)[:, 0], rel_pos / (dist + 1e-6)))

        obs = np.concatenate([
            rel_pos, rel_vel, jp,
            [self.hp[0]], [self.hp[1]],
            [facing], [dist],
            [p1[2]], [p2[2]],
        ]).astype(np.float32)
        return obs

    def _get_opp_obs(self):
        """Observation for r2 (opponent)."""
        d = self.data
        p1 = d.xpos[self.pelvis_ids[0]]
        p2 = d.xpos[self.pelvis_ids[1]]
        rel_pos = (p1 - p2).astype(np.float32)
        rel_vel = (d.cvel[self.pelvis_ids[0]][:3] - d.cvel[self.pelvis_ids[1]][:3]).astype(np.float32)

        jp = np.zeros(29, dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"r2_{name}")
            if jid >= 0:
                jp[i] = d.qpos[self.model.jnt_qposadr[jid]] - self.default_pos[i]

        dist = float(np.linalg.norm(rel_pos))
        R = np.eye(3).flatten()
        mujoco.mju_quat2Mat(R, d.qpos[self.r2_qpos_off + 3:self.r2_qpos_off + 7])
        facing = float(np.dot(R.reshape(3,3)[:, 0], rel_pos / (dist + 1e-6)))

        obs = np.concatenate([
            rel_pos, rel_vel, jp,
            [self.hp[1]], [self.hp[0]],
            [facing], [dist],
            [p2[2]], [p1[2]],
        ]).astype(np.float32)
        return obs

    def _update_damage(self):
        """Detect wrist-to-torso contacts and apply damage."""
        self._dmg_dealt = [0.0, 0.0]
        self._dmg_taken = [0.0, 0.0]
        self._hit_detected = [False, False]

        for con in range(self.data.ncon):
            c = self.data.contact[con]
            g1, g2 = c.geom1, c.geom2
            b1 = self.geom_body[g1]
            b2 = self.geom_body[g2]

            for agent, opp in [(0, 1), (1, 0)]:
                fists = self.fist_bodies[agent]
                torsos = self.torso_bodies[opp]
                if b1 in fists and b2 in torsos:
                    self._apply_hit(agent, opp, g1, g2, c)
                elif b2 in fists and b1 in torsos:
                    self._apply_hit(agent, opp, g2, g1, c)

    def _apply_hit(self, agent, opp, fist_geom, torso_geom, contact):
        """Apply damage from a fist-to-torso contact."""
        # Relative velocity at contact
        fb = self.model.geom_bodyid[fist_geom]
        tb = self.model.geom_bodyid[torso_geom]
        fist_vel = self.data.cvel[fb][:3]
        torso_vel = self.data.cvel[tb][:3]
        rel_vel = float(np.linalg.norm(fist_vel - torso_vel))

        # Only count as a hit if relative velocity is high enough
        if rel_vel > 0.5:
            # Check if head hit (contact z-height relative to pelvis)
            opp_pelvis_z = self.data.xpos[self.pelvis_ids[opp]][2]
            contact_z = contact.pos[2]
            is_head = contact_z > opp_pelvis_z + 0.35
            dmg_mult = 2.0 if is_head else 1.0

            if rel_vel > 1.0:
                dmg = min(8.0, rel_vel * 4.0 * dmg_mult)
            else:
                dmg = min(2.0, rel_vel * 1.0 * dmg_mult)

            if dmg > 0:
                self.hp[opp] = max(0.0, self.hp[opp] - dmg)
                self._dmg_dealt[agent] += dmg
                self._dmg_taken[opp] += dmg
                self._hit_detected[agent] = True

    def _pelvis_z(self, agent):
        return float(self.data.xpos[self.pelvis_ids[agent]][2])

    def _opp_action(self):
        """Get opponent's action."""
        if self.opp_policy is not None:
            obs = self._get_opp_obs()
            action, _ = self.opp_policy.predict(obs, deterministic=True)
            return np.clip(action, -1, 1)
        elif self.opponent == "sandbag":
            return np.zeros(ACT_DIM, dtype=np.float32)
        elif self.opponent and self.opponent.startswith("scripted:"):
            # Simple scripted opponent
            profile = self.opponent.split(":")[1]
            return self._scripted_action(profile)
        return np.zeros(ACT_DIM, dtype=np.float32)

    def _scripted_action(self, profile):
        """Simple scripted opponent behavior."""
        t = self.step_count * DT
        if profile == "jabbler":
            # Aggressive: walk forward + punch
            vx = 0.3 + 0.1 * math.sin(t * 2)
            arm = np.zeros(14, dtype=np.float32)
            arm[7] = -0.5 * max(0, math.sin(t * 3))  # right shoulder punch
            arm[10] = -0.8 * max(0, math.sin(t * 3))  # right elbow extend
            return np.concatenate([[vx, 0, 0], arm]).astype(np.float32)
        elif profile == "defender":
            # Passive: stand, occasional jab
            arm = np.zeros(14, dtype=np.float32)
            arm[0] = -0.3; arm[3] = 0.5  # guard up
            arm[7] = -0.3; arm[10] = 0.5
            return np.concatenate([[0, 0, 0], arm]).astype(np.float32)
        return np.zeros(ACT_DIM, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._place()
        self.ctrl1.last_action = np.zeros(29, dtype=np.float32)
        self.ctrl2.last_action = np.zeros(29, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        vel_cmd = action[:3]
        arm_residuals = action[3:]

        # Scale velocity command
        vel_cmd_scaled = vel_cmd * np.array([0.5, 0.3, 1.0])

        # Scale arm residuals
        arm_scale = 1.5
        arm_targets = self.default_pos[ARM_INDICES] + arm_residuals * arm_scale

        # r1: walker + arm override
        self.ctrl1.lin_vel_x = float(vel_cmd_scaled[0])
        self.ctrl1.lin_vel_y = float(vel_cmd_scaled[1])
        self.ctrl1.ang_vel_z = float(vel_cmd_scaled[2])
        targets1 = self.ctrl1.step()
        # Override arms
        for i, idx in enumerate(ARM_INDICES):
            targets1[idx] = arm_targets[i]
        for i, aid in enumerate(self.ctrl1.actuator_ids):
            if aid >= 0:
                self.data.ctrl[aid] = targets1[i]

        # r2: opponent
        opp_action = self._opp_action()
        opp_vel = opp_action[:3] * np.array([0.5, 0.3, 1.0])
        opp_arm = self.default_pos[ARM_INDICES] + opp_action[3:] * arm_scale

        self.ctrl2.lin_vel_x = float(opp_vel[0])
        self.ctrl2.lin_vel_y = float(opp_vel[1])
        self.ctrl2.ang_vel_z = float(opp_vel[2])
        targets2 = self.ctrl2.step()
        for i, idx in enumerate(ARM_INDICES):
            targets2[idx] = opp_arm[i]
        for i, aid in enumerate(self.ctrl2.actuator_ids):
            if aid >= 0:
                self.data.ctrl[aid] = targets2[i]

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # Damage detection
        self._update_damage()

        # Reward (RoboStriker weights)
        reward = self._combat_reward(0, action)

        # Termination
        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = False
        if self.hp[0] <= KO_HP or self.hp[1] <= KO_HP:
            terminated = True
        if z0 < FALL_Z or z1 < FALL_Z:
            terminated = True
        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            # Terminal reward
            if self.hp[1] <= 0 or z1 < FALL_Z:
                reward += 5.0
            elif self.hp[0] <= 0 or z0 < FALL_Z:
                reward -= 5.0

        info = {"hp_0": self.hp[0], "hp_1": self.hp[1],
                "dmg_dealt": self._dmg_dealt[0], "dmg_taken": self._dmg_taken[0]}
        return self._get_obs(), float(reward), terminated, truncated, info

    def _combat_reward(self, agent, action):
        """Combat reward using RoboStriker weights."""
        opp = 1 - agent
        reward = 0.0

        # Hit reward (w=50)
        if self._hit_detected[agent]:
            reward += 50.0 * (self._dmg_dealt[agent] / 8.0)

        # Defense penalty (w=8)
        if self._dmg_taken[agent] > 0:
            reward -= 8.0 * (self._dmg_taken[agent] / 8.0)

        # Delta striking force (w=0.3)
        reward += 0.3 * (self._dmg_dealt[agent] - self._dmg_taken[agent])

        # Facing alignment (w=1.2, σ=0.5)
        d = self.data
        p1 = d.xpos[self.pelvis_ids[agent]]
        p2 = d.xpos[self.pelvis_ids[opp]]
        rel = p2 - p1
        dist = float(np.linalg.norm(rel))
        face_dir = rel / (dist + 1e-6)
        R = np.eye(3).flatten()
        qoff = 0 if agent == 0 else self.r2_qpos_off
        mujoco.mju_quat2Mat(R, d.qpos[qoff + 3:qoff + 7])
        facing = float(np.dot(R.reshape(3, 3)[:, 0], face_dir))
        reward += 1.2 * np.exp(-max(0.0, 1.0 - facing) / 0.5)

        # Distance reward (w=1.5, velocity-gated)
        if dist > 0.3:
            vel = d.cvel[self.pelvis_ids[agent]][:3]
            approach = max(0.0, float(np.dot(vel, face_dir)))
            if approach > 0.8:
                reward += 1.5 * np.exp(-dist / 1.0)

        # Fall penalty
        z = self._pelvis_z(agent)
        if z < 0.5:
            reward -= 5.0 * (0.5 - z)

        return reward
