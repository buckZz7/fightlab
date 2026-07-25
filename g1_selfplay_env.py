"""G1 self-play boxing environment.

Two G1 humanoids in an arena, each driven by a frozen LocoBase29 balance
policy + learned arm residuals (the fight layer). The fight policy controls
14 arm joints per agent; the base policy keeps the robot upright.

Wraps the two-agent arena as a single-agent gymnasium.Env for SB3 PPO
training. The opponent is a frozen policy (or random).

Architecture:
  - Arena: g1_arena.build_arena() (two G1s, fist collision, walls)
  - Balance: LocoBase29 per agent (frozen ONNX, runs at 50 Hz)
  - Fight policy: PPO over 14-dim arm residuals, per agent
  - Damage: fist-to-torso contact force, HP-based
  - Termination: knockdown (pelvis below threshold) or timeout

Based on G1PunchEnv (single-robot bag env) and SelfPlayEnv (old boxing env).
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import joblib

from g1_arena import build_arena, SKILL_JOINTS, N_SKILL, N_QPOS, N_QVEL, DT, FRAME_SKIP, RESIDUAL_SCALE
from loco_base29 import LocoBase29, HOME
from loco_base29_rot import LocoBase29Rot

# Per-agent qpos/qvel slice offsets in the combined model
# Robot 1: qpos[0:36], qvel[0:35]
# Robot 2: qpos[36:72], qvel[35:70]
QPOS_OFFSET = [0, N_QPOS]
QVEL_OFFSET = [0, N_QVEL]
ACT_OFFSET = [0, 29]  # 29 actuators per robot

# Torso geom name for damage detection (the main torso body)
TORSO_BODY = "torso_link"

# Observation: own 14 skill joints pos/vel (28) + own torso orientation/omega (7)
#            + own HP (1) + opp HP (1)
#            + opponent relative position (3) + opponent torso orientation (4)
#            + opponent arm joint positions (14) — to track their guard
# Total: 28 + 7 + 2 + 3 + 4 + 14 = 58
OBS_DIM = 58

# Action: 14 arm residuals + 3 walk commands (vx, vy, wz) = 17
N_CMD = 3
ACT_DIM = 17

# HP
MAX_HP = 100.0
DAMAGE_PER_HIT = 15.0      # per unit contact force
DAMAGE_COOLDOWN = 0.3      # seconds between scoring hits (anti-spam)
DAMAGE_CAP_PER_HIT = 25.0  # max HP lost in one hit

# Mocap 23-DoF -> G1 29-DoF arm joint mapping (from g1_mocap_punch_env)
# Only arm joints: 23dof L arm 13-17 -> 29dof 15,16,17,18,20
#                  23dof R arm 18-22 -> 29dof 22,23,24,25,27
MOCAP_ARM_MAP = {
    15: 13, 16: 14, 17: 15, 18: 16, 20: 17,   # left arm (29dof -> 23dof)
    22: 18, 23: 19, 24: 20, 25: 21, 27: 22,   # right arm
}
# 29dof arm joints that have mocap data (10 of 14)
MOCAP_JOINTS_29 = sorted(MOCAP_ARM_MAP.keys())
# The 4 arm joints NOT in mocap (wrist pitch+yaw): 19,21,26,28


class MocapOpponent:
    """Replays mocap boxing arm trajectories as opponent actions.

    This is the RoboStriker warmup stage: a standing opponent that throws
    real boxing punches (from retargeted mocap), giving the fight policy
    something meaningful to dodge, block, and counter.

    Not a neural network — just a looped replay of mocap arm targets
    converted to 14-dim residual space.
    """

    def __init__(self, mocap_path="mocap/kungfu_retargeted/Horse-stance_punch.pkl"):
        d = joblib.load(mocap_path)
        clip = d[list(d.keys())[0]]
        self.dof = clip["dof"]      # (T, 23) retargeted joint angles
        self.fps = int(clip["fps"])  # 30
        self.T = len(self.dof)
        self.frame = 0
        # Control runs at 50Hz, mocap at 30fps -> advance frame every ~1.67 steps
        self._frame_every = 50.0 / self.fps  # 1.67
        self._step_accum = 0.0

    def reset(self):
        self.frame = 0
        self._step_accum = 0.0

    def get_action(self):
        """Returns 17-dim action: [0:14] arm residual, [14:17] walk cmd (zeros)."""
        f = min(self.frame, self.T - 1)

        action = np.zeros(N_SKILL + N_CMD)

        # 14-dim action maps to 29dof joints [15:29] (L arm 15-21, R arm 22-28)
        for i, j29 in enumerate(range(15, 29)):
            if j29 in MOCAP_ARM_MAP:
                j23 = MOCAP_ARM_MAP[j29]
                mocap_angle = self.dof[f, j23]
                home_angle = HOME[j29]
                residual = (mocap_angle - home_angle) / max(RESIDUAL_SCALE[i], 0.01)
                action[i] = np.clip(residual, -1, 1)

        # Advance frame
        self._step_accum += 1.0
        if self._step_accum >= self._frame_every:
            self.frame = (self.frame + 1) % self.T  # loop
            self._step_accum = 0.0

        return action


class G1SelfPlayEnv(gym.Env):
    """Single-agent view of the G1 boxing arena for PPO training.

    The challenger (agent 1) is trained; the opponent (agent 2) is frozen.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_model=None, opponent_mocap=False,
                 max_steps=2000, randomize=True):
        super().__init__()
        self.model = build_arena()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

        # Actuator ctrlrange for clipping
        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Body/geom IDs for each agent
        self._setup_ids()

        # Domain randomization base values
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

        # Balance policies (one per robot, both forward-facing)
        self.loco = [LocoBase29(), LocoBase29()]

        # Opponent: neural net, mocap replay, or random
        self.opponent = opponent_model  # SB3 model or None
        self.mocap_opp = MocapOpponent() if opponent_mocap else None

        # Action/obs spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)

        # State
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}

    def _setup_ids(self):
        """Cache body/geom IDs for both agents."""
        self.pelvis_id = []
        self.torso_id = []
        self.fist_geoms = []  # [agent][left, right] geom ids

        for i, pfx in enumerate(["r1_", "r2_"]):
            self.pelvis_id.append(self.model.body(f"{pfx}pelvis").id)
            self.torso_id.append(self.model.body(f"{pfx}{TORSO_BODY}").id)
            fists = []
            for side in ("left", "right"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"{pfx}{side}_fist_col")
                fists.append(gid)
            self.fist_geoms.append(fists)

        # Torso subtree body IDs — hardcoded list of body names that are
        # valid punch targets (torso + head + arms). Static list avoids the
        # O(nbody * depth) parent walk that was hanging __init__.
        # Valid punch targets: torso + head + arms (full upper body)
        TORSO_SUBTREE_NAMES = [
            TORSO_BODY,  # torso_link
            "head_link",  # head (if it exists as separate body)
            "left_shoulder_pitch_link", "left_shoulder_roll_link",
            "left_shoulder_yaw_link", "left_elbow_link",
            "left_wrist_roll_link", "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link",
            "right_shoulder_yaw_link", "right_elbow_link",
            "right_wrist_roll_link", "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]
        self.torso_bodies = []
        for i, pfx in enumerate(["r1_", "r2_"]):
            body_ids = set()
            for name in TORSO_SUBTREE_NAMES:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"{pfx}{name}")
                if bid >= 0:
                    body_ids.add(bid)
            self.torso_bodies.append(body_ids)

    def _get_obs(self, agent=0):
        """Observation for the given agent."""
        qp_off = QPOS_OFFSET[agent]
        qv_off = QVEL_OFFSET[agent]
        opp = 1 - agent

        # Own skill joints (14 pos + 14 vel)
        q = self.data.qpos[qp_off + 7 + 15: qp_off + 7 + 29]
        qd = self.data.qvel[qv_off + 6 + 15: qv_off + 6 + 29]

        # Own torso orientation (quat 4 + angular vel 3)
        quat = self.data.qpos[qp_off + 3: qp_off + 7]
        omega = self.data.qvel[qv_off + 3: qv_off + 6]

        # HP
        hp_self = np.array([self.hp[agent]])
        hp_opp = np.array([self.hp[opp]])

        # Opponent relative position (3) + orientation (4)
        my_pelvis = self.data.xpos[self.pelvis_id[agent]]
        opp_pelvis = self.data.xpos[self.pelvis_id[opp]]
        rel_pos = opp_pelvis - my_pelvis
        opp_quat = self.data.qpos[QPOS_OFFSET[opp] + 3: QPOS_OFFSET[opp] + 7]

        # Opponent arm joint positions (14) — to read their guard
        opp_arms = self.data.qpos[QPOS_OFFSET[opp] + 7 + 15: QPOS_OFFSET[opp] + 7 + 29]

        return np.concatenate([
            q, qd, quat, omega, hp_self, hp_opp,
            rel_pos, opp_quat, opp_arms
        ]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set both robots standing at HOME pose
        for agent in range(2):
            off = QPOS_OFFSET[agent]
            self.data.qpos[off + 2] = 0.75          # pelvis height
            self.data.qpos[off + 7: off + 36] = HOME
            self.data.qpos[off + 7: off + 36] += self.np_random.uniform(-0.02, 0.02, 29)
            # Both robots face forward (no rotation). The balance policy
            # can't handle a 180° rotation. The fight policy learns to
            # approach and strike from any angle — the mocap opponent's
            # arms still punch forward, which is fine for warmup.
            # Robots placed at 0.3m apart so punches can reach.

        if self.randomize:
            self.model.body_mass[:] = self._base_mass * self.np_random.uniform(
                0.95, 1.05, self._base_mass.shape)
            self.model.geom_friction[:] = self._base_friction * self.np_random.uniform(
                0.9, 1.1)

        mujoco.mj_forward(self.model, self.data)
        for loco in self.loco:
            loco.reset()
        if self.mocap_opp is not None:
            self.mocap_opp.reset()

        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}

        return self._get_obs(0), {}

    def _pelvis_z(self, agent):
        return float(self.data.xpos[self.pelvis_id[agent]][2])

    def _opp_action(self, agent):
        """Get action for the opponent (agent 1)."""
        if self.mocap_opp is not None:
            return self.mocap_opp.get_action()
        if self.opponent is None:
            return np.zeros(ACT_DIM)  # random stands still (better than random noise)
        obs = self._get_obs(agent)
        a, _ = self.opponent.predict(obs, deterministic=True)
        return np.clip(a, -1, 1)

    def step(self, action):
        """Step the arena. action = challenger (agent 0) 17-dim:
        [0:14] = arm residuals, [14:17] = walk cmd (vx, vy, wz)
        """
        # Split action: arm residuals + walk commands
        arm_action = np.clip(action[:N_SKILL], -1, 1)
        walk_cmd = np.clip(action[N_SKILL:], -1, 1)
        # Scale walk commands: vx in [-0.5, 0.5] m/s, vy [-0.3, 0.3], wz [-1, 1] rad/s
        walk_scaled = walk_cmd * np.array([0.5, 0.3, 1.0])

        # Get opponent action
        opp_action = self._opp_action(1)
        if self.mocap_opp is not None:
            opp_arm = opp_action
            opp_walk = np.zeros(3)  # mocap opponent stands still
        elif self.opponent is not None:
            opp_arm = opp_action[:N_SKILL]
            opp_walk = opp_action[N_SKILL:] * np.array([0.5, 0.3, 1.0])
        else:
            opp_arm = opp_action
            opp_walk = np.zeros(3)

        actions = [arm_action, opp_arm]

        # Set walk commands on balance policies
        self.loco[0].set_command(walk_scaled[0], walk_scaled[1], walk_scaled[2])
        self.loco[1].set_command(opp_walk[0], opp_walk[1], opp_walk[2])

        # Low-pass filter both agents' arm residuals
        for agent in range(2):
            raw = actions[agent][:N_SKILL] * RESIDUAL_SCALE
            self._residuals[agent] += 0.25 * (raw - self._residuals[agent])

        # Physics step loop (same as G1PunchEnv)
        for _ in range(self.frame_skip):
            for agent in range(2):
                qp = QPOS_OFFSET[agent]
                qv = QVEL_OFFSET[agent]
                self.loco[agent].update(self.data.qpos[qp:qp+N_QPOS],
                                        self.data.qvel[qv:qv+N_QVEL])
                target = self.loco[agent].target.copy()
                target[SKILL_JOINTS] += self._residuals[agent]

                # Compute PD torque for this agent's 29 actuators
                tau = self.loco[agent].pd_torque(
                    self.data.qpos[qp:qp+N_QPOS],
                    self.data.qvel[qv:qv+N_QVEL],
                    target_override=target)
                act_off = ACT_OFFSET[agent]
                self.data.ctrl[act_off:act_off+29] = np.clip(
                    tau, self.lo[act_off:act_off+29], self.hi[act_off:act_off+29])
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1

        # Damage detection: check fist-to-torso contacts
        self._update_damage()

        # Reward for challenger (agent 0)
        reward = self._compute_reward(0)

        # Termination
        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps

        # Win/loss shaping at episode end
        if terminated or truncated:
            if self.hp[1] <= 0 or (z1 < 0.4 and z0 > 0.4):
                reward += 25.0   # win
            elif self.hp[0] <= 0 or (z0 < 0.4 and z1 > 0.4):
                reward -= 25.0   # loss

        info = {
            "hp_0": self.hp[0], "hp_1": self.hp[1],
            "pelvis_z_0": z0, "pelvis_z_1": z1,
        }
        return self._get_obs(0), reward, terminated, truncated, info

    def _update_damage(self):
        """Detect fist-to-opponent-torso contacts and apply HP damage.

        Anti-shove: a hit only scores if the fist has positive relative
        velocity toward the opponent (RoboStriker's hit reward condition).
        Shoving (body translation, zero relative fist velocity) scores zero.
        Also requires facing the opponent (facing penalty, Son & Kwon 2023).
        """
        for con in range(self.data.ncon):
            contact = self.data.contact[con]
            g1, g2 = contact.geom1, contact.geom2
            # Get contact force via mj_contactForce (6-dim wrench)
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, con, f)
            force = np.linalg.norm(f[:3])

            if force < 5.0:
                continue  # ignore light brush contacts

            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]

            for attacker in range(2):
                defender = 1 - attacker
                for fist in self.fist_geoms[attacker]:
                    fist_body = self.model.geom_bodyid[fist]
                    hit_registered = False
                    if g1 == fist and b2 in self.torso_bodies[defender]:
                        hit_registered = True
                    elif g2 == fist and b1 in self.torso_bodies[defender]:
                        hit_registered = True

                    if hit_registered:
                        # Anti-shove: compute relative fist velocity
                        # v_rel = v_fist_body - v_torso_attacker, projected
                        # onto attack direction (toward opponent)
                        fist_body_id = self.model.geom_bodyid[fist]
                        fist_vel = self.data.cvel[fist_body_id][3:6].copy()
                        torso_id = self.torso_id[attacker]
                        torso_vel = self.data.cvel[torso_id][3:6].copy()
                        rel_vel = fist_vel - torso_vel

                        # Attack direction: from attacker torso toward defender torso
                        attack_dir = (
                            self.data.xpos[self.torso_id[defender]] -
                            self.data.xpos[torso_id]
                        )
                        attack_norm = np.linalg.norm(attack_dir)
                        if attack_norm > 1e-6:
                            attack_dir /= attack_norm

                        # Relative velocity projected onto attack direction
                        punch_speed = float(np.dot(rel_vel, attack_dir))

                        # Facing check: are the robots facing each other?
                        # Use pelvis forward direction (x-axis of pelvis frame)
                        pelvis_quat = self.data.qpos[
                            QPOS_OFFSET[attacker] + 3: QPOS_OFFSET[attacker] + 7
                        ]
                        # Pelvis forward = rotate [1,0,0] by pelvis quaternion
                        pw, px, py, pz = pelvis_quat
                        forward = np.array([
                            1 - 2*(py*py + pz*pz),
                            2*(px*py + pw*pz),
                            2*(px*pz - pw*py),
                        ])
                        facing = float(np.dot(forward, attack_dir))

                        # Hit only scores if:
                        # 1. Punch speed > 0.5 m/s (fist moving toward opponent)
                        # 2. Facing > 0 (roughly facing opponent)
                        # 3. Force > 5N (already checked above)
                        if punch_speed > 0.5 and facing > 0:
                            self._register_hit(
                                attacker, defender, force, punch_speed)
                        elif punch_speed <= 0.5:
                            # Shove: record but don't score (for logging)
                            self._contact_states[(attacker, defender)] = {
                                'force': force, 'damage': 0.0,
                                'punch_speed': punch_speed,
                                'shove': True}

    def _register_hit(self, attacker, defender, force, punch_speed=0.0):
        """Record a hit with cooldown and HP damage.

        Damage scales with both force AND punch speed — a fast, clean
        punch hurts more than a slow push.
        """
        t = self.step_count * DT * FRAME_SKIP
        if t - self._last_hit_time[attacker] > DAMAGE_COOLDOWN:
            # Damage = force * speed bonus (fast punches do more damage)
            speed_mult = min(2.0, 1.0 + punch_speed * 0.5)
            dmg = min(DAMAGE_CAP_PER_HIT,
                      force * DAMAGE_PER_HIT / 100.0 * speed_mult)
            self.hp[defender] = max(0, self.hp[defender] - dmg)
            self._last_hit_time[attacker] = t
            self._contact_states[(attacker, defender)] = {
                'force': force, 'damage': dmg,
                'punch_speed': punch_speed, 'shove': False}

    def _compute_reward(self, agent):
        """Reward for the given agent.

        Based on RoboStriker + Son&Kwon reward design:
        - Hit reward requires relative fist velocity (anti-shove)
        - Facing penalty prevents back-attack degenerate mode
        - Punch speed weighted: fast clean punches score more
        """
        opp = 1 - agent
        reward = 0.0

        # Damage dealt (positive) — weighted by punch quality
        if (agent, opp) in self._contact_states:
            cs = self._contact_states[(agent, opp)]
            if cs.get('shove', False):
                # Shove: no reward (anti-shove mechanism)
                pass
            else:
                reward += 5.0 * cs['damage']
                # Big hit bonus: clean fast punches
                if cs['force'] > 50 and cs.get('punch_speed', 0) > 1.0:
                    reward += 5.0  # quality strike bonus
                elif cs['force'] > 50:
                    reward += 2.0  # forceful contact

        # Damage taken (negative) — strong penalty for getting hit
        if (opp, agent) in self._contact_states:
            cs = self._contact_states[(opp, agent)]
            if not cs.get('shove', False):
                reward -= 3.0 * cs['damage']  # defensive penalty

        # Stability: small alive bonus, big tilt penalty
        z = self._pelvis_z(agent)
        quat = self.data.qpos[QPOS_OFFSET[agent] + 3: QPOS_OFFSET[agent] + 7]
        tilt = 1.0 - float(quat[0]) ** 2
        reward += 0.05 if z > 0.6 else 0.0
        reward -= 3.0 * tilt
        reward -= 2.0 * max(0.0, 0.72 - z)

        # Facing penalty (w=10, doubled): must face opponent to score.
        # Prevents back-attack / hold-down-from-behind degenerate behavior.
        pelvis_quat = self.data.qpos[
            QPOS_OFFSET[agent] + 3: QPOS_OFFSET[agent] + 7
        ]
        pw, px, py, pz = pelvis_quat
        forward = np.array([
            1 - 2*(py*py + pz*pz),
            2*(px*py + pw*pz),
            2*(px*pz - pw*py),
        ])
        my_pos = self.data.xpos[self.pelvis_id[agent]]
        opp_pos = self.data.xpos[self.pelvis_id[opp]]
        to_opp = opp_pos - my_pos
        to_opp_norm = np.linalg.norm(to_opp)
        facing = 0.0
        if to_opp_norm > 1e-6:
            to_opp /= to_opp_norm
            facing = float(np.dot(forward, to_opp))
            # Strong penalty for not facing opponent
            if facing < 0:
                reward += 10.0 * facing  # -10 when fully turned away
            elif facing < 0.5:
                reward += 5.0 * (facing - 0.5)  # -2.5 at 0 facing

        # Approach maintenance: reward staying close to opponent
        # and penalize running away
        dist = to_opp_norm if to_opp_norm > 1e-6 else 0
        if facing > 0 and dist < 0.5:
            reward += 1.0  # engaged bonus
        if facing > 0 and dist < 0.2:
            reward += 1.0  # close quarters bonus
        # Distance-based approach (only when facing)
        if facing > 0:
            reward += 0.5 * max(0.0, 1.0 - dist)

        # Energy penalty
        reward -= 0.01 * float(np.sum(np.square(self._residuals[agent])))

        # Clear contact states after reward computation
        self._contact_states = {}

        return reward


def make_g1_selfplay_env(opponent_path=None, **kw):
    from stable_baselines3 import PPO
    opp = PPO.load(opponent_path) if opponent_path else None
    return G1SelfPlayEnv(opponent_model=opp, **kw)
