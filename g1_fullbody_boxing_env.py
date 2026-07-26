"""Track B training env: full-body G1 boxing, trained IN our scene.

Design (per the sim2sim-transfer lesson):
  - Train from scratch IN scene_29dof.xml (only stable path; borrowed
    ONNX/mocap brains faceplant here - verified).
  - G1 Moves clips are REWARD REFERENCES (motion-match bonus), never controllers.
  - Configurable so we can later warm-start into other arenas/attack types
    WITHOUT a rebuild:  arena_shape (square/octagon/open), allow_kicks (bool).
  - Boxing rules (punch-only, no kicks) enforced now; flips to
    kick-enabled when allow_kicks=True.

Action space: 29-DoF residuals? No - full-body PPO acts on the SAME 17-dim
the frozen base used (arm residuals + walk cmd) BUT on a full-body walker
that actually balances (SB3-trained, stable 30s - proven by Gen1-3).
Wait - Gen1-3 were stable BUT could not strike. So we keep the 17-dim
interface and re-reward for striking; the walker is already stable.

Actually the cleanest: agent action = 17-dim [arm resid 14 + walk 3], same as
g1_selfplay_env, but the REWARD now (a) requires real fist-to-torso
contact with relative velocity > 0.5 (anti-shove, already there),
(b) ADDS a G1-Moves motion-match bonus so the policy learns PUNCH
SHAPE not just "approach and face", (c) balance penalty if pelvis drops.

This reuses the proven-stable walker + existing damage/HP/KO/rules.
"""
import os, sys, gymnasium as gym
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from g1_arena import build_arena
from loco_base29 import StandPD, HOME, KP, KD

DT = 0.01
FRAME_SKIP = 4          # 50 Hz control, 200 Hz sim -> wait, scene uses dt=0.005
# scene_29dof sets opt.timestep; we step frame_skip times per action.
N_SKILL = 14
N_CMD = 3
ACT_DIM = N_SKILL + N_CMD
OBS_DIM = 58
MAX_HP = 100.0
TORSO_BODY = "torso_link"
RESIDUAL_SCALE = 0.15

# G1 Moves punch reference clips (joint trajectories, 29-DoF)
# Loaded once; used as a motion-match bonus target for the arms.
def _load_move_refs(motion_dir):
    refs = {}
    for name, fn in [("jab", "M_ShortMove12_quickjab"),
                     ("lowpunch", "M_Move2_lowpunch"),
                     ("rapidpunch", "M_Move7_rapidpunch")]:
        p = os.path.join(motion_dir, "motion", f"{fn}.npz")
        if os.path.exists(p):
            m = np.load(p)
            refs[name] = {
                "arm": m["joint_pos"].astype(np.float32)[:, 15:29],  # arm joints only
                "fps": float(m["fps"]),
            }
    return refs


class G1FullBodyBoxingEnv(gym.Env):
    """Full-body boxing. Agent = r1 (trained); r2 = frozen opponent.

    Swappable later via constructor flags (warm-start ready):
      arena_shape: 'square' (now) | 'octagon' | 'open'
      allow_kicks: False (boxing now) | True (later)
    """
    metadata = {"render_modes": []}

    def __init__(self, opponent_model=None, opponent_mocap=False,
                 max_steps=2000, randomize=True,
                 arena_shape="square", allow_kicks=False,
                 motion_dir=None):
        super().__init__()
        self.model = build_arena(shape=arena_shape) if hasattr(build_arena, "_shape_ok") else build_arena()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.005
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize
        self.arena_shape = arena_shape
        self.allow_kicks = allow_kicks
        self.motion_dir = motion_dir or os.path.join(
            os.path.dirname(__file__), "g1moves")

        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()

        # Balance bases (proven-stable SB3 walkers live in the policy; these
        # Balance substrate: StandPD (stable PD-to-HOME, no ONNX).
        # The ONNX balance base is UNSTABLE (verified falls 2.8-13.4s).
        self.loco = [StandPD(), StandPD()]

        # Opponent
        self.opponent = opponent_model
        self.opponent2 = None
        self.mocap_opp = MocapOpponent() if opponent_mocap else None

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)

        # Move refs (G1 Moves punch shapes) for motion-match bonus
        self.move_refs = _load_move_refs(self.motion_dir)
        self._active_ref = None
        self._ref_t = 0.0

        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._last_hit_time = [-1.0, -1.0]
        self._contact_states = {}
        self._setup_ids()
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()

    # ---- IDs (mirrors g1_selfplay_env) ----
    def _setup_ids(self):
        self.pelvis_id = []
        self.torso_id = []
        self.fist_geoms = []
        self.torso_bodies = []
        TORSO_SUBTREE = ["torso_link", "head_link", "waist_yaw_link",
                         "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                         "left_elbow_link", "right_elbow_link"]
        for i, pfx in enumerate(["r1_", "r2_"]):
            self.pelvis_id.append(self.model.body(f"{pfx}pelvis").id)
            self.torso_id.append(self.model.body(f"{pfx}{TORSO_BODY}").id)
            fg = []
            for side in ("left", "right"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"{pfx}{side}_fist_sphere")
                if gid >= 0:
                    fg.append(gid)
            self.fist_geoms.append(fg)
            bodies = set()
            for name in TORSO_SUBTREE:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}{name}")
                if bid >= 0:
                    bodies.add(bid)
            self.torso_bodies.append(bodies)

    def _pelvis_z(self, agent):
        return float(self.data.xpos[self.pelvis_id[agent]][2])

    def _robot_bodies(self, agent):
        out = set()
        for b in range(self.model.nbody):
            if self.model.body(b).name.startswith(f"r{agent+1}_"):
                out.add(b)
        return out

    # ---- obs / step mirror g1_selfplay_env, plus move-ref tracking ----
    def _get_obs(self, agent=0):
        # Per-robot qpos/qvel offsets: r1 at 0, r2 at +7.
        off = 0 if agent == 0 else 7
        # own skill joints (arm 14) + torso ori + hp + opp relative + contact
        q = self.data.qpos[off + 7 + 15: off + 7 + 29]      # arm joints 15:29
        qd = self.data.qvel[off + 6 + 15: off + 6 + 29]
        quat = self.data.qpos[off + 3: off + 7]            # root quat
        omega = self.data.qvel[off + 3: off + 6]             # root ang vel
        hp_self = np.array([self.hp[agent]])
        hp_opp = np.array([self.hp[1 - agent]])
        my_pel = self.data.xpos[self.pelvis_id[agent]]
        opp_pel = self.data.xpos[self.pelvis_id[1 - agent]]
        rel_pos = opp_pel - my_pel
        opp_quat = self.data.qpos[7 + 3: 7 + 7]  # r2 (opp) quat at +7
        # facing: dot of my forward (x-axis of torso) with rel_pos dir
        R = self._quat_to_rot(quat)
        fwd = R[:, 0]
        dist = float(np.linalg.norm(rel_pos))
        facing = float(np.dot(fwd, rel_pos) / (dist + 1e-6))
        return np.concatenate([
            q, qd, quat, omega, hp_self, hp_opp,
            rel_pos, opp_quat, np.array([facing, dist]),
            self._residuals[agent],
        ]).astype(np.float64)

    def _quat_to_rot(self, q):
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    def _opp_action(self, agent):
        if self.mocap_opp is not None:
            return self.mocap_opp.get_action()
        if self.opponent2 is not None and agent == 1:
            obs = self._get_obs(1)
            a, _ = self.opponent2.predict(obs, deterministic=True)
            return np.clip(a, -1, 1)
        if self.opponent is None:
            return np.zeros(ACT_DIM)
        obs = self._get_obs(agent)
        a, _ = self.opponent.predict(obs, deterministic=True)
        return np.clip(a, -1, 1)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        # place bots: r1 at -0.6, r2 at 0.3 (square ring spacing)
        for ai, (x, pfx) in enumerate([(-0.6, "r1_"), (0.3, "r2_")]):
            off = ai * 7                      # free-joint qpos offset (r1:0, r2:7)
            self.data.qpos[off : off + 3] = [x, 0, 0.793]   # root x,y,z (world)
            self.data.qpos[off + 3 : off + 29] = HOME[3:]   # 26 joint targets
        if self.randomize:
            self._randomize()
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._contact_states = {}
        self._active_ref = None
        self._ref_t = 0.0
        return self._get_obs(0), {}

    def _randomize(self):
        # mass +/- 10%, friction +/- 15% (stay-transferable; NOT geometry)
        self.model.body_mass[:] = self._base_mass * np.random.uniform(0.9, 1.1, self.model.nbody)
        self.model.geom_friction[:, 0] = self._base_friction[:, 0] * np.random.uniform(0.85, 1.15, self.model.ngeom)

    def step(self, action):
        arm_action = np.clip(action[:N_SKILL], -1, 1)
        walk_cmd = np.clip(action[N_SKILL:], -1, 1)
        walk_scaled = walk_cmd * np.array([0.5, 0.3, 1.0])
        opp_action = self._opp_action(1)
        if self.mocap_opp is not None:
            opp_arm = opp_action
            opp_walk = np.zeros(3)
        elif self.opponent is not None:
            opp_arm = opp_action[:N_SKILL]
            opp_walk = opp_action[N_SKILL:] * np.array([0.5, 0.3, 1.0])
        else:
            opp_arm = opp_action
            opp_walk = np.zeros(3)

        actions = [arm_action, opp_arm]
        self.loco[0].set_command(walk_scaled[0], walk_scaled[1], walk_scaled[2])
        self.loco[1].set_command(opp_walk[0], opp_walk[1], opp_walk[2])

        for _ in range(self.frame_skip):
            for agent in range(2):
                raw = actions[agent][:N_SKILL] * RESIDUAL_SCALE
                self._residuals[agent] += 0.25 * (raw - self._residuals[agent])
            for agent in range(2):
                qp = agent * 7   # free-joint qpos offset: r1=0, r2=7
                self.loco[agent].update(self.data.qpos, self.data.qvel, off=qp)
                target = self.loco[agent].target.copy()
                target[15:29] += self._residuals[agent]  # arm residuals
                tau = self.loco[agent].pd_torque(
                    self.data.qpos, self.data.qvel, off=qp,
                    target_override=target)
                act_off = agent * 29
                self.data.ctrl[act_off:act_off+29] = np.clip(tau, self.lo[act_off:act_off+29], self.hi[act_off:act_off+29])
            mujoco.mj_step(self.model, self.data, 1)

        self.step_count += 1
        self._update_damage()
        reward = self._compute_reward(0)

        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            if self.hp[1] <= 0 or (z1 < 0.4 and z0 > 0.4):
                reward += 25.0
            elif self.hp[0] <= 0 or (z0 < 0.4 and z1 > 0.4):
                reward -= 25.0
        info = {"hp_0": self.hp[0], "hp_1": self.hp[1],
                "pelvis_z_0": z0, "pelvis_z_1": z1}
        return self._get_obs(0), reward, terminated, truncated, info

    # ---- damage + reward (boxing: fist-to-torso only) ----
    def _update_damage(self):
        self._contact_states = {}
        for con in range(self.data.ncon):
            c = self.data.contact[con]
            g1, g2 = c.geom1, c.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]
            for agent, opp in [(0, 1), (1, 0)]:
                fists = self.fist_geoms[agent]
                if g1 in fists or g2 in fists:
                    other_b = b2 if g1 in fists else b1
                    if other_b in self.torso_bodies[opp]:
                        # anti-shove: fist must move toward opp with rel vel > 0.5
                        fist_body = self.model.geom_bodyid[g1] if g1 in fists else self.model.geom_bodyid[g2]
                        rel_vel = self._fist_rel_vel(agent, opp)
                        shove = rel_vel < 0.5
                        if not shove:
                            dmg = min(8.0, max(0.0, rel_vel * 4.0))
                            self.hp[opp] = max(0.0, self.hp[opp] - dmg)
                            self._contact_states[(agent, opp)] = {
                                "shove": False, "dmg": dmg}

    def _fist_rel_vel(self, agent, opp):
        fist_body = self.fist_geoms[agent][0] if self.fist_geoms[agent] else 0
        fb = self.model.geom_bodyid[fist_body]
        opp_pel = self.data.xpos[self.pelvis_id[opp]]
        fist_pos = self.data.xpos[fb]
        # body-frame velocity of fist
        fist_vel = self.data.cvel[fb][:3]
        off = 7 if agent == 1 else 0
        R = self._quat_to_rot(self.data.qpos[off + 3: off + 7])
        rel = opp_pel - fist_pos
        rel_dir = rel / (np.linalg.norm(rel) + 1e-6)
        return float(np.dot(fist_vel, rel_dir))

    def _compute_reward(self, agent=0):
        opp = 1 - agent
        reward = 0.0
        # damage dealt (PRIMARY)
        if (agent, opp) in self._contact_states:
            cs = self._contact_states[(agent, opp)]
            if not cs.get("shove", False):
                reward += 15.0 * (cs.get("dmg", 0.0) / 8.0)
        # facing: must face opponent (boxing requires facing)
        my_fwd = self._quat_to_rot(self.data.qpos[3:7])[:, 0]
        rel = self.data.xpos[self.pelvis_id[opp]] - self.data.xpos[self.pelvis_id[agent]]
        dist = np.linalg.norm(rel)
        facing = np.dot(my_fwd, rel / (dist + 1e-6))
        reward += 1.0 * max(0.0, facing)          # face opponent
        reward += 0.3 * (1.0 - min(1.0, abs(dist - 0.5)))  # engage at ~0.5m
        reward -= 0.05 * max(0.0, 0.4 - self._pelvis_z(agent))  # balance
        return float(reward)

    def render(self, height=480, width=640):
        rend = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 3.4; cam.elevation = -8; cam.lookat[:] = [0.45, 0, 0.8]
        rend.update_scene(self.data, camera=cam)
        return rend.render()


# Minimal mocap opponent stub (kept for interface compat)
class MocapOpponent:
    def get_action(self):
        return np.zeros(ACT_DIM)


def make_g1_fullbody_env(opponent_path=None, opponent_path2=None, **kw):
    from stable_baselines3 import PPO
    opp = PPO.load(opponent_path) if opponent_path else None
    opp2 = PPO.load(opponent_path2) if opponent_path2 else None
    env = G1FullBodyBoxingEnv(opponent_model=opp, opponent_model2=opp2, **kw)
    return env


if __name__ == "__main__":
    import time
    e = G1FullBodyBoxingEnv(max_steps=500, randomize=False)
    o, _ = e.reset()
    print("obs dim", o.shape, "act dim", e.action_space.shape)
    t0 = time.time()
    for i in range(500):
        a = e.action_space.sample()
        o, r, term, trunc, info = e.step(a)
        if term or trunc:
            break
    print(f"ran {i+1} steps in {time.time()-t0:.1f}s; hp={e.hp}")
