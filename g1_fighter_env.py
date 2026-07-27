"""Track B fighter env: balance substrate + punches.

Stacks on top of a TRAINED balance policy (models/balance_v1.zip).
The balance policy (29-dim PD residual) is FROZEN and provides
standing. The fighter policy adds:
  - action (17): [arm residual 14 | walk cmd 3]
  - arm residuals modulate the frozen balance's arm targets
    (learns punches via motion-match bonus)
  - walk cmd drives footwork (lean/step via PD on legs)
  - reward: damage dealt (anti-shove) + facing + approach
    + motion-match bonus (clean punch shape) + balance penalty

r2 = frozen balance stander (or a loaded opponent for bout mode).

This is INDEPENDENT of the running balance training -- it
just needs balance_v1.zip to exist when we launch it.
"""
import os, sys, glob, math
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym

from street_arena import build_default_2bot
from loco_base29 import StandPD, KP, KD, HOME
from g1_moves_reward import MoveCoach

DT = 0.002   # MUST match g1_arena.DT (RK4 stable timestep). 0.01 sagged.
FRAME_SKIP = 1   # control every physics step (500Hz). Mirrors g1_balance_env.
N_SKILL = 14
N_CMD = 3
ACT_DIM = N_SKILL + N_CMD
OBS_DIM = 85  # quat4 + angvel3 + jrel29 + jvel29 + hp_self1 +
              # hp_opp1 + rel3 + pelvis_z1 + residuals14 (SEE _get_obs)
MAX_HP = 100.0
RESIDUAL_SCALE = 0.15
NATIVE_ROOT_X = [-0.6, 0.3]


class G1FighterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, balance_path=None, opponent_path=None,
                 max_steps=1500, randomize=True, motion_dir=None, demo=False,
                 king=None, ring="ropes"):
        super().__init__()
        self.model = build_default_2bot()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize
        # demo mode: apply walk as visible leg motion + larger arm
        # residual scale so scripted punches/footwork actually SHOW.
        # Never set by training (fighters learn small residuals).
        self.demo = demo
        # king: which robot (0 or 1) is the reigning king -> RED gloves.
        # None -> default r1=red, r2=blue (legacy left/right).
        self.king = king
        # No glove coloring (bare-handed street fight).

        # Capture fighters use HOME as the balance base (NOT self.native).
        # The balance policy is TRAINED relative to HOME (g1_balance_env:
        # target = HOME + act*SCALE_BAL, obs jrel = qp - HOME). The frozen
        # policy must receive the exact same obs encoding + target base or
        # it falls. self.native (XML default) != HOME -> mismatch.
        mujoco.mj_resetData(self.model, self.data)
        self.native = HOME.copy()   # balance base == HOME (aligned with training)

        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()

        # balance substrate (frozen) + coach (motion match)
        self.balance = self._load_ppo(balance_path) if balance_path else None
        self.coach = MoveCoach(motion_dir or os.path.join(
            os.path.dirname(__file__), "g1moves"))
        self.opponent = self._load_ppo(opponent_path) if opponent_path else None

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)

        self._setup_ids()
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._contact_states = {}
        # DR defaults (overridden by _randomize() when randomize=True).
        # Must exist even when not randomized, or _apply_control crashes.
        self.rng_kp = KP.copy()
        self.rng_kd = KD.copy()
        self.torque_noise_std = 0.0          # no noise unless randomized
        self._delay_buf = {}

    def _color_gloves_by_king(self, king):
        """Recolor fist geoms: the KING robot gets RED gloves, the
        challenger gets BLUE. king in {0,1}. r1 -> red if king==0
        else blue; r2 -> red if king==1 else blue."""
        RED = [0.95, 0.12, 0.12, 1.0]
        BLUE = [0.15, 0.35, 0.95, 1.0]
        for i in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if nm.endswith("_fist_col"):
                slot = 0 if nm.startswith("r1_") else 1
                self.model.geom_rgba[i] = RED if slot == king else BLUE

    def _load_ppo(self, path):
        from stable_baselines3 import PPO
        return PPO.load(path)

    def _setup_ids(self):
        self.pelvis_id = []
        self.fist_geoms = []
        self.torso_bodies = []
        self.leg_bodies = []
        self.head_bodies = []
        # Upper body targets (full damage) — head excluded (2x damage)
        TORSO = ["torso_link",
                 "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                 "left_shoulder_roll_link", "right_shoulder_roll_link",
                 "left_shoulder_yaw_link", "right_shoulder_yaw_link",
                 "left_elbow_link", "right_elbow_link",
                 "left_wrist_roll_link", "right_wrist_roll_link",
                 "left_wrist_pitch_link", "right_wrist_pitch_link",
                 "left_wrist_yaw_link", "right_wrist_yaw_link",
                 "waist_yaw_link", "waist_roll_link",
                 "left_hip_pitch_link", "right_hip_pitch_link",
                 "left_hip_roll_link", "right_hip_roll_link",
                 "left_hip_yaw_link", "right_hip_yaw_link"]
        # Head = 2x damage target
        HEAD = ["head_link"]
        # Leg targets (reduced damage)
        LEGS = ["left_knee_link", "right_knee_link"]
        for i, pfx in enumerate(["r1_", "r2_"]):
            self.pelvis_id.append(self.model.body(f"{pfx}pelvis").id)
            # Weapons = WRIST collision geoms (punches) + ANKLE collision geoms (kicks)
            # Full combat: fists and feet both count as striking weapons.
            fg = []
            for side in ("left", "right"):
                for body_name in ("wrist_yaw_link", "ankle_roll_link"):
                    wb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                            f"{pfx}{side}_{body_name}")
                    if wb >= 0:
                        for j in range(self.model.body_geomnum[wb]):
                            gid = self.model.body_geomadr[wb] + j
                            if self.model.geom_contype[gid] > 0:
                                fg.append(("wrist" in body_name, gid))  # (is_fist, geom_id)
            self.fist_geoms.append(fg)
            bodies = set()
            for nm in TORSO:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{nm}")
                if bid >= 0:
                    bodies.add(bid)
            self.torso_bodies.append(bodies)
            heads = set()
            for nm in HEAD:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{nm}")
                if bid >= 0:
                    heads.add(bid)
            self.head_bodies.append(heads)
            legs = set()
            for nm in LEGS:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{nm}")
                if bid >= 0:
                    legs.add(bid)
            self.leg_bodies.append(legs)

    def _pelvis_z(self, a):
        return float(self.data.xpos[self.pelvis_id[a]][2])

    def _place(self):
        mujoco.mj_resetData(self.model, self.data)
        for ai, x in enumerate(NATIVE_ROOT_X):
            off = ai * 36            # each robot = 7 (root) + 29 (joints) = 36 qpos
            self.data.qpos[off:off + 3] = [x, 0, 0.793]   # native G1 height
            # Face each other: r1 (left, ai=0) faces +X, r2 (right, ai=1) faces -X.
            # 180° rotation around Z axis (vertical) = quat [0,0,0,1] = faces -X
            # without flipping upside down (Y rotation would flip the robot).
            if ai == 0:
                self.data.qpos[off + 3:off + 7] = [1, 0, 0, 0]   # face +X (toward r2)
            else:
                self.data.qpos[off + 3:off + 7] = [0, 0, 0, 1]   # face -X (toward r1)
            self.data.qpos[off + 7:off + 36] = self.native[:29]  # HOME joints (29)
        if self.randomize:
            self._randomize()
        mujoco.mj_forward(self.model, self.data)

    def _randomize(self):
        # --- Sim2Real domain randomization (canonical G1 set) ---
        # Mass + payload
        self.model.body_mass[:] = self._base_mass * np.random.uniform(0.9, 1.1, self.model.nbody)
        # Foot/contact friction
        self.model.geom_friction[:, 0] = self._base_friction[:, 0] * np.random.uniform(0.85, 1.15, self.model.ngeom)
        # PD-gain jitter (so policy isn't overfit to nominal gains)
        self.rng_kp = KP * np.random.uniform(0.85, 1.15, 29)
        self.rng_kd = KD * np.random.uniform(0.85, 1.15, 29)
        # Actuator torque-noise std (applied each step)
        self.torque_noise_std = 0.05 * np.mean(np.abs(self.lo))  # ~5% of ctrl range

    def _apply_control(self, agent_tau, agent_ctrl_slice):
        """Apply PD torque with sim2real noise: torque noise + 1-step delay."""
        if self.torque_noise_std > 0:
            noise = np.random.normal(0.0, self.torque_noise_std, agent_tau.shape)
            tau = agent_tau + noise
        else:
            tau = agent_tau  # deterministic: no noise when std=0
        # 1-step actuator delay (ring buffer): command lags 1 control step
        if not hasattr(self, "_delay_buf"):
            self._delay_buf = {}
        buf = self._delay_buf.setdefault(agent_ctrl_slice.start,
                                         np.zeros_like(tau))
        out = buf.copy()
        buf[:] = np.clip(tau, self.lo[agent_ctrl_slice], self.hi[agent_ctrl_slice])
        self.data.ctrl[agent_ctrl_slice] = out

    def _get_obs(self, agent=0):
        off = 0 if agent == 0 else 36
        qp = self.data.qpos[off:off + 36]
        # qvel stride is 35 (free joint = 6 vel DOF). qpos_off - 1 = qvel_off.
        qv_off = (off - 1) if off > 0 else 0
        qv = self.data.qvel[qv_off:qv_off + 35]
        quat = qp[3:7]
        angvel = qv[3:6]
        jrel = qp[7:36] - self.native
        jvel = qv[6:35]
        hp_self = np.array([self.hp[agent]])
        hp_opp = np.array([self.hp[1 - agent]])
        rel = self.data.xpos[self.pelvis_id[1 - agent]] - self.data.xpos[self.pelvis_id[agent]]
        return np.concatenate([quat, angvel, jrel, jvel, hp_self, hp_opp,
                            rel, np.array([self._pelvis_z(agent)]),
                            self._residuals[agent]]).astype(np.float64)

    def reset(self, seed=None, options=None):
        self._place()
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._contact_states = {}
        self.coach.reset()
        return self._get_obs(0), {}

    def _bal_obs(self, agent):
        """obs for the frozen balance policy (65-dim: quat,angvel,jrel,jvel)."""
        off = 0 if agent == 0 else 36
        qp = self.data.qpos[off:off + 36]
        # qvel stride is 35 per robot (free joint = 6 vel DOF, not 7):
        # r1 qvel[0:35], r2 qvel[35:70]. qpos_off - 1 = qvel_off.
        qv_off = (off - 1) if off > 0 else 0
        qv = self.data.qvel[qv_off:qv_off + 35]
        return np.concatenate([qp[3:7], qv[3:6], qp[7:36] - self.native, qv[6:35]]).astype(np.float64)

    def _opp_action(self):
        if self.opponent is None:
            return np.zeros(ACT_DIM)
        o, _ = self.opponent.predict(self._get_obs(1), deterministic=True)
        return np.clip(o, -1, 1)

    def _update_damage(self):
        self._contact_states = {}
        self._dmg_dealt = [0.0, 0.0]
        self._dmg_taken = [0.0, 0.0]
        for con in range(self.data.ncon):
            c = self.data.contact[con]
            g1, g2 = c.geom1, c.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            for agent, opp in [(0, 1), (1, 0)]:
                weapons = self.fist_geoms[agent]
                for is_fist, wgid in weapons:
                    if g1 == wgid or g2 == wgid:
                        other = b2 if g1 == wgid else b1
                        is_leg = other in self.leg_bodies[opp]
                        is_torso = other in self.torso_bodies[opp]
                        if not (is_torso or is_leg):
                            break
                        rel_vel = self._weapon_rel_vel(agent, opp, wgid)
                        # Determine hit height: head = high on torso
                        # The G1 has no separate head body — use contact z.
                        opp_pelvis_z = self.data.xpos[self.pelvis_id[opp]][2]
                        contact_z = self.data.contact[con].pos[2]
                        is_head = is_torso and (contact_z > opp_pelvis_z + 0.35)
                        # Damage multipliers:
                        # - Head (high hit): 2.0x (devastating, incentivizes blocking)
                        # - Punch to body: 1.0x | Kick to body: 1.5x
                        # - Kick to legs: 0.5x | Punch to legs: 0.3x
                        if is_head:
                            dmg_mult = 2.0
                        elif is_leg:
                            dmg_mult = 0.5 if not is_fist else 0.3
                        else:
                            dmg_mult = 1.0 if is_fist else 1.5
                        if rel_vel > 1.0:
                            dmg = min(8.0, max(0.0, rel_vel * 4.0 * dmg_mult))
                        elif rel_vel > 0.5:
                            dmg = min(2.0, rel_vel * 1.0 * dmg_mult)
                        else:
                            dmg = 0.0
                        if dmg > 0:
                            self.hp[opp] = max(0.0, self.hp[opp] - dmg)
                            self._dmg_dealt[agent] += dmg
                            self._dmg_taken[opp] += dmg
                            self._contact_states[(agent, opp)] = {"shove": dmg == 0, "dmg": dmg}
                        break

    def _weapon_rel_vel(self, agent, opp, wgid):
        fb = self.model.geom_bodyid[wgid]
        opp_pel = self.data.xpos[self.pelvis_id[opp]]
        fist_pos = self.data.xpos[fb]
        fist_vel = self.data.cvel[fb][:3]
        off = 36 if agent == 1 else 0
        R = self._quat_to_rot(self.data.qpos[off + 3:off + 7])
        rel = opp_pel - fist_pos
        rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
        return float(np.dot(fist_vel, rel_dir))

    def _fist_rel_vel(self, agent, opp):
        """Legacy: uses first weapon geom."""
        wgid = self.fist_geoms[agent][0][1] if isinstance(self.fist_geoms[agent][0], tuple) else self.fist_geoms[agent][0]
        return self._weapon_rel_vel(agent, opp, wgid)

    def _foot_contact_count(self, agent=0):
        """Count a bot's foot geoms touching a non-self body (the floor)."""
        pfx = "r1_" if agent == 0 else "r2_"
        fb = {pfx + "left_ankle_roll_link", pfx + "right_ankle_roll_link",
              pfx + "left_ankle_pitch_link", pfx + "right_ankle_pitch_link"}
        n = 0
        for c in range(self.data.ncon):
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[self.data.contact[c].geom1])
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   self.model.geom_bodyid[self.data.contact[c].geom2])
            if (b1 in fb and b2 not in fb) or (b2 in fb and b1 not in fb):
                n += 1
        return n

    def _quat_to_rot(self, q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    def _walk_legs(self, walk_cmd, t, off):
        """Convert (vx,vy,wz) walk cmd -> leg-joint target deltas.

        Legs are joints 7:15 (8 joints: L/R hip_pitch, hip_roll,
        knee, ankle_pitch, ankle_roll). A simple alternating
        stepping gait: hip_pitch + knee swing out of phase L/R,
        wz pivots the hips, vx scales stride. Returns 8-vec.
        Used by demo footwork (and is the real footwork path the
        walk-cmd reward was always expecting but never wired).
        """
        vx, vy, wz = walk_cmd
        s = math.sin(t * 2.5)
        c = math.sin(t * 2.5 + math.pi)  # opposite leg phase
        stride = 0.5 * vx
        swing = 0.6 * abs(vx)
        # target[0:12] = qpos[7:19] = L leg (0:6) + R leg (6:12):
        #   [L_hip_p, L_hip_r, L_hip_y, L_knee, L_ank_p, L_ank_r,
        #    R_hip_p, R_hip_r, R_hip_y, R_knee, R_ank_p, R_ank_r]
        d = np.zeros(12)
        d[0] = stride * s          # L hip pitch
        d[3] = swing * max(0.0, -s) + 0.25  # L knee lifts
        d[6] = stride * c          # R hip pitch
        d[9] = swing * max(0.0, -c) + 0.25  # R knee lifts
        d[1] = 0.15 * vy           # L hip roll (lateral)
        d[7] = -0.15 * vy          # R hip roll
        d[4] = -0.25 * wz         # L ankle pivot
        d[10] = 0.25 * wz          # R ankle pivot
        return d

    def step(self, action):
        arm_action = np.clip(action[:N_SKILL], -1, 1)
        walk_cmd = np.clip(action[N_SKILL:], -1, 1)
        walk_scaled = walk_cmd * np.array([0.5, 0.3, 1.0])
        opp_action = self._opp_action()

        arm_scale = 1.5 if self.demo else RESIDUAL_SCALE
        t = self.step_count * DT * self.frame_skip

        # r1 walk -> leg targets (was dead code; now applied)
        leg1 = self._walk_legs(walk_scaled, t, off=0) if self.demo else np.zeros(8)
        # r2 walk (opponent)
        opp_walk = opp_action[N_SKILL:] * np.array([0.5, 0.3, 1.0]) if self.opponent else np.zeros(3)
        leg2 = self._walk_legs(opp_walk, t, off=36) if self.demo else np.zeros(8)

        for _ in range(self.frame_skip):
            # r1: frozen balance residual + arm residual
            bal_act = self.balance.predict(self._bal_obs(0), deterministic=True)[0] if self.balance else np.zeros(29)
            target = bal_act * 0.40 + self.native   # self.native == HOME
            if self.demo:
                # For a watchable demo, let the SCRIPTED guard/punch
                # own the arm joints fully (the smoke balance policy
                # flails its arms and fights the guard). Legs/waist
                # still come from balance so they keep standing.
                target[15:29] = self.native[15:29] + self._residuals[0]
            else:
                target[15:29] += self._residuals[0]   # arm residual -> qpos[22:36]
            if self.demo:
                # balance leg target overpowers small walk deltas;
                # blend walk IN (60%) on the LEG slice target[0:12]
                # (qpos[7:19] = both legs) so footwork shows.
                target[0:12] = target[0:12] * 0.4 + (self.native[0:12] + leg1) * 0.6
            kp = getattr(self, "rng_kp", KP)
            kd = getattr(self, "rng_kd", KD)
            tau1 = kp * (target - self.data.qpos[7:36]) - kd * self.data.qvel[6:35]
            self._apply_control(tau1, slice(0, 29))
            # r2: opponent or stand (no arm action)
            if self.opponent:
                bal_act2 = self.balance.predict(self._bal_obs(1), deterministic=True)[0] if self.balance else np.zeros(29)
                t2 = bal_act2 * 0.40 + self.native
                if self.demo:
                    t2[15:29] = self.native[15:29] + self._residuals[1]
                else:
                    t2[15:29] += self._residuals[1]
                if self.demo:
                    t2[0:12] = t2[0:12] * 0.4 + (self.native[0:12] + leg2) * 0.6
                tau2 = kp * (t2 - self.data.qpos[43:72]) - kd * self.data.qvel[41:70]
                self._apply_control(tau2, slice(29, 58))
            else:
                tau2 = StandPD().pd_torque(self.data.qpos, self.data.qvel, off=36)
                self._apply_control(tau2, slice(29, 58))
            mujoco.mj_step(self.model, self.data, 1)

        # residual update (lerp toward arm action * scale)
        for agent in range(2):
            raw = (arm_action if agent == 0 else opp_action[:N_SKILL]) * arm_scale
            lerp = 0.5 if self.demo else 0.25
            self._residuals[agent] += lerp * (raw - self._residuals[agent])

        self.step_count += 1
        self._update_damage()

        z0 = self._pelvis_z(0)
        # Activate coach when the arm residual is large (bot is throwing);
        # pick the clip whose shape best matches current arm pose.
        arm_res = self._residuals[0]
        arm_qpos = self.data.qpos[7+15:7+29] - HOME[15:29]
        if np.linalg.norm(arm_res) > 0.05:
            if self.coach.active is None:
                best, best_err = None, 1e9
                for nm in self.coach.refs:
                    tgt = self.coach.refs[nm]["arm"][0]
                    err = np.mean((arm_qpos - tgt) ** 2)
                    if err < best_err:
                        best_err, best = err, nm
                self.coach.start(best)
        else:
            self.coach.active = None
        self._coach_bonus = self.coach.step(arm_qpos, DT * self.frame_skip)
        reward = self._compute_reward(0, action)

        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            # RoboStriker terminal: opponent below h_min = win; self below = loss.
            # Use a MODERATE bonus (not the old flat +25 which dwarfed the
            # per-step shaping and biased toward degenerate rush-down). Scale
            # by HP margin so a decisive KO beats a lucky stumble.
            margin = (self.hp[0] - self.hp[1]) / MAX_HP   # +1 (won clean) .. -1
            if self.hp[1] <= 0 or z1 < 0.4:
                reward += 5.0 + 3.0 * max(0.0, margin)   # win
            elif self.hp[0] <= 0 or z0 < 0.4:
                reward -= 5.0 + 3.0 * max(0.0, -margin)  # loss
        info = {"hp_0": self.hp[0], "hp_1": self.hp[1], "pelvis_z_0": z0, "pelvis_z_1": z1}
        return self._get_obs(0), reward, terminated, truncated, info

    def _compute_reward(self, agent=0, action=None):
        opp = 1 - agent
        reward = 0.0
        # --- Strike reward (RoboStriker: w_hit=50, gated force+speed) ---
        if (agent, opp) in self._contact_states:
            cs = self._contact_states[(agent, opp)]
            if not cs.get("shove", False):
                reward += 50.0 * (cs.get("dmg", 0.0) / 8.0)
        # --- Defensive penalty (RoboStriker: w_def=8) ---
        if self._dmg_taken[agent] > 0:
            reward -= 8.0 * (self._dmg_taken[agent] / 8.0)
        # --- Delta striking force (RoboStriker: w_str=0.3) ---
        reward += 0.3 * (self._dmg_dealt[agent] - self._dmg_taken[agent])
        # --- Motion-match bonus: our AMP-equivalent (RoboStriker shows
        #     dropping it drops hit-rate 0.685->0.49) ---
        reward += 2.0 * self._coach_bonus
        # --- Facing alignment (RoboStriker: w_face=1.2, exp falloff) ---
        R = self._quat_to_rot(self.data.qpos[3:7])
        rel = self.data.xpos[self.pelvis_id[opp]] - self.data.xpos[self.pelvis_id[agent]]
        dist = np.linalg.norm(rel)
        face_dir = rel / (dist + 1e-6)
        facing = np.dot(R[:, 0], face_dir)
        reward += 1.2 * np.exp(-max(0.0, 1.0 - facing) / 0.5)
        # --- Velocity-gated approach (RoboStriker: w_dist=1.5, sigma=1.0,
        #     v_th=0.8; reward only when moving TOWARD opponent in range) ---
        vel = self.data.cvel[self.pelvis_id[agent]][:3]  # world linear vel
        approach = max(0.0, np.dot(vel, face_dir))
        in_range = np.exp(-abs(dist - 0.5) / 1.0)
        reward += 1.5 * (1.0 if approach > 0.8 else 0.0) * in_range
        # --- Balance / keep-standing penalty ---
        reward -= 0.05 * max(0.0, 0.4 - self._pelvis_z(agent))
        # --- FOOT PLANT (HoST: both feet down = stable, not hopping) ---
        reward += 0.05 * min(self._foot_contact_count(agent), 2)
        # --- ACTION SMOOTHNESS (HoST: L2 delta-action, prevents jitter punches) ---
        if hasattr(self, "_prev_act"):
            reward -= 0.01 * float(np.sum((action - self._prev_act) ** 2))
        self._prev_act = action.copy()
        return float(reward)

    @property
    def _last_coach_bonus(self):
        return getattr(self, "_coach_bonus", 0.0)

    def render(self, height=480, width=640):
        rend = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]
        rend.update_scene(self.data, camera=cam)
        return rend.render()


if __name__ == "__main__":
    import time
    e = G1FighterEnv(max_steps=200, randomize=False)
    o, _ = e.reset()
    print("obs", o.shape, "act", e.action_space.shape)
    t0 = time.time()
    for i in range(200):
        a = e.action_space.sample()
        o, r, term, trunc, info = e.step(a)
        if term or trunc:
            break
    print(f"ran {i+1} steps in {time.time()-t0:.1f}s; hp={e.hp}")
