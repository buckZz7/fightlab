"""Combat Fine-Tuning: Stage 2 of the FightLab pipeline.

Takes a trained motion tracker (Stage 1) and fine-tunes it with
combat rewards in the 2-bot G1FighterEnv. The tracker provides
physical skills (punching, dodging, balance); combat rewards add
strategic behavior (hit opponent, face opponent, avoid damage).

This produces fighter_v2: a mocap-trained fighter that can actually
box, vs the old fighter_v1 which was pure RL from scratch.

Usage:
  python3 train_combat.py --tracker models/motion_tracker_v2 \
      --steps 1000000 --envs 16 --out models/fighter_v2
"""
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")
import numpy as np
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from g1_fighter_env import G1FighterEnv, N_SKILL, N_CMD, ACT_DIM, DT, FRAME_SKIP
from loco_base29 import StandPD, KP, KD, HOME

MAX_HP = 100.0


class CombatEnv(G1FighterEnv):
    """Fighter env with a frozen motion tracker as the balance substrate
    (instead of PD-to-HOME). The tracker provides natural motion; combat
    rewards add strategy. The policy outputs arm residuals + walk commands
    on top of the tracker."""

    def __init__(self, tracker_path=None, opponent_path=None,
                 max_steps=1500, randomize=True):
        # Don't call G1FighterEnv.__init__ — we override _setup
        super(G1FighterEnv, self).__init__()
        from street_arena import build_default_2bot
        self.model = build_default_2bot()
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = DT
        self.frame_skip = FRAME_SKIP
        self.max_steps = max_steps
        self.randomize = randomize
        self.demo = False
        self.king = None

        # Load the motion tracker as the balance substrate
        self.tracker = self._load_ppo(tracker_path) if tracker_path else None
        # Load opponent (another fighter, or None = PD stand)
        self.opponent = self._load_ppo(opponent_path) if opponent_path else None

        # Action space: same as G1FighterEnv (14 arm skills + 3 walk)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(ACT_DIM,), dtype=np.float32)
        # Match G1FighterEnv obs dim (85)
        from g1_fighter_env import OBS_DIM
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64)

        self._setup_ids()
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()
        self.lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.hi = self.model.actuator_ctrlrange[:, 1].copy()
        from g1_moves_reward import MoveCoach
        motion_dir = os.path.join(os.path.dirname(os.environ.get("G1_SCENE_XML", "")), "motions") if os.environ.get("G1_SCENE_XML") else None
        self.coach = MoveCoach(motion_dir)
        self.hp = [MAX_HP, MAX_HP]
        self._residuals = [np.zeros(N_SKILL), np.zeros(N_SKILL)]
        self._contact_states = {}
        self._dmg_dealt = [0.0, 0.0]
        self._dmg_taken = [0.0, 0.0]
        self._prev_act = np.zeros(ACT_DIM)
        self.native = HOME.copy()

    def _load_ppo(self, path):
        if path is None:
            return None
        try:
            from stable_baselines3 import PPO
            return PPO.load(path)
        except Exception:
            return None

    def _bal_obs(self, agent):
        """obs for the motion tracker (same format as G1FighterEnv)."""
        off = 0 if agent == 0 else 36
        qp = self.data.qpos[off:off + 36]
        qv_off = (off - 1) if off > 0 else 0
        qv = self.data.qvel[qv_off:qv_off + 35]
        return np.concatenate([qp[3:7], qv[3:6], qp[7:36] - self.native, qv[6:35]]).astype(np.float64)

    def _tracker_obs(self, agent):
        """obs for the motion tracker (matches train_motion_tracker format).
        Tracker expects: [joint_pos - home, joint_vel, ref_pos - home, ref_vel]
        In combat we use zero reference (maintain current pose = balance mode)."""
        off = 0 if agent == 0 else 36
        qp = self.data.qpos[off:off + 36]
        qv_off = (off - 1) if off > 0 else 0
        qv = self.data.qvel[qv_off:qv_off + 35]
        jrel = qp[7:36] - self.native  # (29,)
        jvel = qv[6:35]                # (29,)
        # Zero reference = "maintain current pose"
        ref_pos = np.zeros(29)
        ref_vel = np.zeros(29)
        return np.concatenate([jrel, jvel, ref_pos, ref_vel]).astype(np.float32)

    def step(self, action):
        arm_action = np.clip(action[:N_SKILL], -1, 1)
        walk_cmd = np.clip(action[N_SKILL:], -1, 1)
        walk_scaled = walk_cmd * np.array([0.5, 0.3, 1.0])
        opp_action = self._opp_action()

        arm_scale = 1.5  # combat: larger arm residuals for punching
        t = self.step_count * DT * self.frame_skip

        for _ in range(self.frame_skip):
            # r1: motion tracker provides base joints, arm residual on top
            if self.tracker:
                tracker_obs = self._tracker_obs(0)
                tracker_act, _ = self.tracker.predict(tracker_obs, deterministic=True)
                # tracker outputs 29 joint position deltas
                target = self.native + tracker_act * 0.5
            else:
                target = self.native.copy()

            # Add arm residual on top of tracker's arm targets
            target[15:29] += self._residuals[0]

            # Simple walk (override leg targets with walk)
            leg1 = self._walk_legs(walk_scaled, t, off=0)
            target[0:12] = target[0:12] * 0.5 + (self.native[0:12] + leg1) * 0.5

            kp = KP
            kd = KD
            tau1 = kp * (target - self.data.qpos[7:36]) - kd * self.data.qvel[6:35]
            self._apply_control(tau1, slice(0, 29))

            # r2: opponent or PD stand
            if self.opponent:
                if self.tracker:
                    tracker_obs2 = self._tracker_obs(1)
                    tracker_act2, _ = self.tracker.predict(tracker_obs2, deterministic=True)
                    t2 = self.native + tracker_act2 * 0.5
                else:
                    t2 = self.native.copy()
                t2[15:29] += self._residuals[1]
                tau2 = kp * (t2 - self.data.qpos[43:72]) - kd * self.data.qvel[41:70]
                self._apply_control(tau2, slice(29, 58))
            else:
                tau2 = StandPD().pd_torque(self.data.qpos, self.data.qvel, off=36)
                self._apply_control(tau2, slice(29, 58))

            mujoco.mj_step(self.model, self.data, 1)

        # residual update
        for agent in range(2):
            raw = (arm_action if agent == 0 else opp_action[:N_SKILL]) * arm_scale
            self._residuals[agent] += 0.25 * (raw - self._residuals[agent])

        self.step_count += 1
        self._update_damage()

        reward = self._combat_reward(0, action)
        z0 = self._pelvis_z(0)
        z1 = self._pelvis_z(1)
        terminated = z0 < 0.4 or z1 < 0.4
        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            margin = (self.hp[0] - self.hp[1]) / MAX_HP
            if self.hp[1] <= 0 or z1 < 0.4:
                reward += 5.0 + 3.0 * max(0.0, margin)
            elif self.hp[0] <= 0 or z0 < 0.4:
                reward -= 5.0 + 3.0 * max(0.0, -margin)

        info = {"hp_0": self.hp[0], "hp_1": self.hp[1]}
        return self._get_obs(0), float(reward), terminated, truncated, info

    def _combat_reward(self, agent=0, action=None):
        """Combat rewards: hit + face + approach + survive."""
        opp = 1 - agent
        reward = 0.0

        # Strike reward (hit)
        if (agent, opp) in self._contact_states:
            cs = self._contact_states[(agent, opp)]
            if not cs.get("shove", False):
                reward += 50.0 * (cs.get("dmg", 0.0) / 8.0)

        # Defensive penalty
        if self._dmg_taken[agent] > 0:
            reward -= 8.0 * (self._dmg_taken[agent] / 8.0)

        # Delta striking force
        reward += 0.3 * (self._dmg_dealt[agent] - self._dmg_taken[agent])

        # Facing alignment
        R = self._quat_to_rot(self.data.qpos[3:7])
        rel = self.data.xpos[self.pelvis_id[opp]] - self.data.xpos[self.pelvis_id[agent]]
        dist = np.linalg.norm(rel)
        face_dir = rel / (dist + 1e-6)
        facing = np.dot(R[:, 0], face_dir)
        reward += 1.2 * np.exp(-max(0.0, 1.0 - facing) / 0.5)

        # Approach reward
        vel = self.data.cvel[self.pelvis_id[agent]][:3]
        approach = max(0.0, np.dot(vel, face_dir))
        in_range = np.exp(-abs(dist - 0.5) / 1.0)
        reward += 1.5 * (1.0 if approach > 0.8 else 0.0) * in_range

        # Balance penalty
        reward -= 0.05 * max(0.0, 0.4 - self._pelvis_z(agent))

        # Action smoothness
        if hasattr(self, "_prev_act"):
            reward -= 0.01 * float(np.sum((action - self._prev_act) ** 2))
        self._prev_act = action.copy()

        return float(reward)

    def _walk_legs(self, walk_cmd, t, off):
        """Simple alternating gait."""
        vx, vy, wz = walk_cmd
        s = math.sin(t * 2.5)
        c = math.sin(t * 2.5 + math.pi)
        stride = 0.5 * vx
        swing = 0.6 * abs(vx)
        d = np.zeros(12)
        d[0] = stride * s
        d[3] = swing * max(0.0, -s) + 0.25
        d[6] = stride * c
        d[9] = swing * max(0.0, -c) + 0.25
        d[1] = 0.15 * vy
        d[7] = -0.15 * vy
        d[4] = -0.25 * wz
        d[10] = 0.25 * wz
        return d


def make_env(tracker_path, opponent_path, max_steps, seed):
    def _init():
        env = CombatEnv(tracker_path=tracker_path, opponent_path=opponent_path,
                       max_steps=max_steps, randomize=True)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, help="motion tracker model path")
    ap.add_argument("--opponent", default=None, help="opponent fighter (None=PD stand)")
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--out", default="models/fighter_v2")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=1500)
    a = ap.parse_args()

    env = SubprocVecEnv([make_env(a.tracker, a.opponent, a.max_steps, i)
                         for i in range(a.envs)])

    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                verbose=1)

    print(f"[combat] training {a.steps} steps with {a.envs} envs (tracker={a.tracker})...")
    model.learn(total_timesteps=a.steps)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    model.save(a.out)
    print(f"[combat] saved {a.out}")


if __name__ == "__main__":
    main()
