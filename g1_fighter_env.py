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
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym

from g1_arena import build_arena
from loco_base29 import StandPD, KP, KD, HOME
from g1_moves_reward import MoveCoach

DT = 0.01
FRAME_SKIP = 4
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
                 max_steps=1500, randomize=True, motion_dir=None):
        super().__init__()
        self.model = build_arena(ring="ropes", half=2.4)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize

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

    def _load_ppo(self, path):
        from stable_baselines3 import PPO
        return PPO.load(path)

    def _setup_ids(self):
        self.pelvis_id = []
        self.fist_geoms = []
        self.torso_bodies = []
        TORSO = ["torso_link", "head_link", "left_shoulder_pitch_link",
                  "right_shoulder_pitch_link", "left_elbow_link", "right_elbow_link"]
        for i, pfx in enumerate(["r1_", "r2_"]):
            self.pelvis_id.append(self.model.body(f"{pfx}pelvis").id)
            fg = []
            for side in ("left", "right"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"{pfx}{side}_fist_col")
                if gid >= 0:
                    fg.append(gid)
            self.fist_geoms.append(fg)
            bodies = set()
            for nm in TORSO:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{nm}")
                if bid >= 0:
                    bodies.add(bid)
            self.torso_bodies.append(bodies)

    def _pelvis_z(self, a):
        return float(self.data.xpos[self.pelvis_id[a]][2])

    def _place(self):
        mujoco.mj_resetData(self.model, self.data)
        for ai, x in enumerate(NATIVE_ROOT_X):
            off = ai * 7
            self.data.qpos[off:off + 3] = [x, 0, 0.793]
            self.data.qpos[off + 3:off + 32] = HOME[:29]  # 29 joint targets
        if self.randomize:
            self._randomize()
        mujoco.mj_forward(self.model, self.data)

    def _randomize(self):
        self.model.body_mass[:] = self._base_mass * np.random.uniform(0.9, 1.1, self.model.nbody)
        self.model.geom_friction[:, 0] = self._base_friction[:, 0] * np.random.uniform(0.85, 1.15, self.model.ngeom)

    def _get_obs(self, agent=0):
        off = 0 if agent == 0 else 7
        qp = self.data.qpos[off:off + 36]
        qv = self.data.qvel[off:off + 35]
        quat = qp[3:7]
        angvel = qv[3:6]
        jrel = qp[7:36] - HOME
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
        off = 0 if agent == 0 else 7
        qp = self.data.qpos[off:off + 36]
        qv = self.data.qvel[off:off + 35]
        return np.concatenate([qp[3:7], qv[3:6], qp[7:36] - HOME, qv[6:35]]).astype(np.float64)

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
                fists = self.fist_geoms[agent]
                if g1 in fists or g2 in fists:
                    other = b2 if g1 in fists else b1
                    if other in self.torso_bodies[opp]:
                        rel_vel = self._fist_rel_vel(agent, opp)
                        # RoboStriker gates hits on BOTH relative speed AND
                        # contact force. We use rel_vel as the speed gate
                        # (force via mj_contactForce is heavier; rel_vel is a
                        # robust proxy for a real strike vs a shove).
                        if rel_vel > 1.0:          # real punch (forceful)
                            dmg = min(8.0, max(0.0, rel_vel * 4.0))
                        elif rel_vel > 0.5:        # glancing
                            dmg = min(2.0, rel_vel * 1.0)
                        else:                      # shove (no reward)
                            dmg = 0.0
                        if dmg > 0:
                            self.hp[opp] = max(0.0, self.hp[opp] - dmg)
                            self._dmg_dealt[agent] += dmg
                            self._dmg_taken[opp] += dmg
                            self._contact_states[(agent, opp)] = {"shove": dmg == 0, "dmg": dmg}

    def _fist_rel_vel(self, agent, opp):
        fb = self.model.geom_bodyid[self.fist_geoms[agent][0]]
        opp_pel = self.data.xpos[self.pelvis_id[opp]]
        fist_pos = self.data.xpos[fb]
        fist_vel = self.data.cvel[fb][:3]
        off = 7 if agent == 1 else 0
        R = self._quat_to_rot(self.data.qpos[off + 3:off + 7])
        rel = opp_pel - fist_pos
        rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
        return float(np.dot(fist_vel, rel_dir))

    def _quat_to_rot(self, q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    def step(self, action):
        arm_action = np.clip(action[:N_SKILL], -1, 1)
        walk_cmd = np.clip(action[N_SKILL:], -1, 1)
        walk_scaled = walk_cmd * np.array([0.5, 0.3, 1.0])
        opp_action = self._opp_action()

        actions = [arm_action, opp_action[:N_SKILL] if len(opp_action) > N_SKILL else opp_action]
        opp_walk = opp_action[N_SKILL:] * np.array([0.5, 0.3, 1.0]) if self.opponent else np.zeros(3)

        for _ in range(self.frame_skip):
            for agent in range(2):
                raw = actions[agent][:N_SKILL] * RESIDUAL_SCALE
                self._residuals[agent] += 0.25 * (raw - self._residuals[agent])
            # r1: frozen balance residual + arm residual
            bal_act = self.balance.predict(self._bal_obs(0), deterministic=True)[0] if self.balance else np.zeros(29)
            target = bal_act * 0.40 + HOME   # MUST match SCALE_BAL in g1_balance_env (0.40), else substrate starved
            target[15:29] += self._residuals[0]
            tau1 = KP * (target - self.data.qpos[7:36]) - KD * self.data.qvel[6:35]
            self.data.ctrl[:29] = np.clip(tau1, self.lo[:29], self.hi[:29])
            # r2: opponent or stand (no arm action)
            if self.opponent:
                bal_act2 = self.balance.predict(self._bal_obs(1), deterministic=True)[0] if self.balance else np.zeros(29)
                t2 = bal_act2 * 0.40 + HOME
                t2[15:29] += self._residuals[1]
                tau2 = KP * (t2 - self.data.qpos[14:43]) - KD * self.data.qvel[13:42]
                self.data.ctrl[29:58] = np.clip(tau2, self.lo[29:], self.hi[29:])
            else:
                tau2 = StandPD().pd_torque(self.data.qpos, self.data.qvel, off=7)
                self.data.ctrl[29:58] = np.clip(tau2, self.lo[29:], self.hi[29:])
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1
        self._update_damage()
        # motion-match bonus (clean punch shape) -- coach tracks active punch.
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
        reward = self._compute_reward(0)

        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            # RoboStriker terminal: opponent below h_min = win; self below = loss
            if self.hp[1] <= 0:
                reward += 25.0
            elif self.hp[0] <= 0:
                reward -= 25.0
            elif z1 < 0.4 and z0 > 0.4:
                reward += 25.0
            elif z0 < 0.4 and z1 > 0.4:
                reward -= 25.0
        info = {"hp_0": self.hp[0], "hp_1": self.hp[1], "pelvis_z_0": z0, "pelvis_z_1": z1}
        return self._get_obs(0), reward, terminated, truncated, info

    def _compute_reward(self, agent=0):
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
        # --- Velocity-gated approach (RoboStriker: w_dist=1.5) ---
        # reward only when moving TOWARD opponent (anti passive/spam)
        vel = self.data.cvel[self.pelvis_id[agent]][:3]  # world linear vel
        approach = max(0.0, np.dot(vel, face_dir))
        reward += 1.5 * (1.0 if approach > 0.1 else 0.0) * np.exp(-abs(dist - 0.5) / 1.0)
        # --- Balance / keep-standing penalty ---
        reward -= 0.05 * max(0.0, 0.4 - self._pelvis_z(agent))
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
