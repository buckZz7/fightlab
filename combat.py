"""Combat rules engine + ShadowBoxer opponents for the FightLab
King-of-the-Hill league.

Part 1 — CombatJudge / rules engine:
  - Legal targets: torso + head (front and side)
  - Legal weapons: fists, feet, knees, elbows (full combat)
  - Round structure: 3 rounds x ROUND_SECONDS, bell, rest between rounds
  - Scoring: 10-point must system (round winner gets 10, loser 9 or less)
    with deductions for fouls. HP/KO is a stoppage, not the only win path.
  - Win conditions: KO/TKO (HP<=0 or knockdown), decision (after 3 rounds).

Part 2 — ShadowBoxer: scripted combatant (punches, kicks, dodges, guards,
footwork) used as a reference opponent / demo bot.

Full combat: punches, kicks, spins all count. Any clean hit with
sufficient relative velocity = damage.
"""
import os, sys, math
import json
import numpy as np

# ===========================================================================
# combat_rules.py — rules engine
# ===========================================================================
ROUNDS = 3
ROUND_SECONDS = 30.0
REST_SECONDS = 10.0
KO_HP = 0.0
KNOCKDOWN_Z = 0.45  # pelvis height below this = knockdown
FOUL_DISQUALIFY_POINTS = 4.0  # cumulative foul deductions -> DQ
FOUL_DEDUCTION = 1.0  # points lost per foul (per round cap)

# Legal contact: fist geom must touch a legal target body of the opponent.
# The env already restricts damage to torso_bodies + head_link; we additionally
# require the contact geom to be a fist (enforced in env._update_damage) and
# block rear-of-head via facing check (attacker must face defender).


class CombatJudge:
    """Tracks a single bout under combat rules. Drives env step + scores."""

    def __init__(self, env, round_seconds=ROUND_SECONDS, rounds=ROUNDS):
        self.env = env
        self.round_seconds = round_seconds
        self.rounds = rounds
        self.reset()

    def reset(self):
        self.round = 0
        self.round_time = 0.0
        self.scores = [0.0, 0.0]          # cumulative judge points
        self.round_scores = [[], []]      # per-round points per fighter
        self.foul_points = [0.0, 0.0]     # cumulative foul deductions
        self.round_fouls = [0, 0]
        self.ko = False
        self.winner = None
        self.dq = None
        self._fell = False
        self._last_hp = [100.0, 100.0]
        self._last_z = [0.78, 0.78]

    def _legal_contact_only(self, agent, opp):
        """Verify the contact that produced damage was a legal fist strike.

        The env sets self._contact_states[(attacker,defender)] with keys
        'shove' (bool) and 'damage'. A legal punch has shove=False.
        We additionally reject rear-of-head / back hits: the env's facing
        check already requires facing>0, so any scored hit is front-facing.
        """
        cs = self.env._contact_states.get((agent, opp))
        if cs is None:
            return False
        if cs.get('shove', False):
            return False  # illegal: push, not punch
        return True

    def _robot_bodies(self, agent):
        """Return set of body ids belonging to robot `agent` (0 or 1)."""
        # Build from fist_geoms + torso_bodies + all geoms parented to this
        # robot's bodies. Cheap cache.
        if not hasattr(self, '_rb_cache'):
            self._rb_cache = [set(), set()]
            pfx = self.env.prefix[agent] if hasattr(self.env, 'prefix') else ('r1_' if agent == 0 else 'r2_')
            for i in range(self.env.model.nbody):
                name = self.env.model.body(i).name
                if name.startswith(pfx):
                    self._rb_cache[agent].add(i)
        return self._rb_cache[agent]

    def _detect_foul(self, agent, opp):
        """Detect illegal contact (clinch/shove, NOT kicks or punches).

        Full combat: fists (wrists) and feet (ankles) are legal weapons.
        Only penalize body-to-body contact that isn't a strike.
        """
        fouled = False
        for con in range(self.env.data.ncon):
            c = self.env.data.contact[con]
            g1, g2 = c.geom1, c.geom2
            b1 = self.env.model.geom_bodyid[g1]
            b2 = self.env.model.geom_bodyid[g2]
            if (b1 in self._robot_bodies(agent) and b2 in self._robot_bodies(opp)) or \
               (b2 in self._robot_bodies(agent) and b1 in self._robot_bodies(opp)):
                # Check if either geom is a legal weapon (fist or foot)
                weapons = self.env.fist_geoms[agent]
                is_weapon = False
                for is_fist, wgid in weapons:
                    if g1 == wgid or g2 == wgid:
                        is_weapon = True
                        break
                if not is_weapon:
                    fouled = True
        return fouled

    def step(self, actions):
        """Advance one env step under rules. Returns (obs, rew, done, info)."""
        obs, rew, term, trunc, info = self.env.step(actions)

        # Round clock
        self.round_time += self.env.model.opt.timestep * self.env.frame_skip

        # Foul detection (simplified)
        for a in range(2):
            if self._detect_foul(a, 1 - a):
                self.round_fouls[a] += 1
                self.foul_points[a] += FOUL_DEDUCTION
                if self.foul_points[a] >= FOUL_DISQUALIFY_POINTS:
                    self.dq = a
                    self.winner = 1 - a
                    info['disqualification'] = a

        # KO / TKO check
        for a in range(2):
            if self.env.hp[a] <= KO_HP:
                self.ko = True
                self.winner = 1 - a
                info['ko'] = a
            z = self.env._pelvis_z(a)
            if z < KNOCKDOWN_Z:
                # knockdown: award round + standing counts as TKO risk
                self.env.hp[a] = max(0, self.env.hp[a] - 5)  # knockdown damage
                if self.env.hp[a] <= KO_HP:
                    self.ko = True
                    self.winner = 1 - a
                    info['tko'] = a
            # FALL (pelvis below FALL threshold) = round/combat loss even
            # if HP not depleted. The env terminates on fall; judge must
            # record the winner as the bot still on its feet.
            if z < 0.40 and self.winner is None:
                self.winner = 1 - a
                self._fell = True
                info['fall'] = a

        # Round end
        if self.round_time >= self.round_seconds and not self.ko and self.dq is None:
            self._score_round()
            self.round += 1
            self.round_time = 0.0
            self.round_fouls = [0, 0]
            if self.round >= self.rounds:
                self._decide_decision()
                info['decision'] = True
                return obs, rew, True, False, info
            # reset positions for next round (keep HP)
            info['round_end'] = self.round

        done = term or trunc or self.ko or self.dq is not None
        # If the env truncated (max_steps reached) before all rounds played,
        # force a decision now so the bout always yields a scored card.
        if trunc and self.winner is None and not self.ko and self.dq is None:
            # score any in-progress round, then decide
            self._score_round() if self.round_time > 0 else None
            self._decide_decision()
            info['decision'] = True
            done = True
        return obs, rew, term, trunc, info

    def _score_round(self):
        """10-point must system: round winner gets 10, loser 9 (or less).
        Draw if damage difference < DRAW_THRESHOLD (no coin-flip wins)."""
        DRAW_THRESHOLD = 2.0  # less than 2 HP difference = draw round (10-10)
        # Compare legal damage dealt this round
        dmg = [100 - self.env.hp[a] for a in range(2)]
        dmg_delta = [dmg[a] - dmg[1 - a] for a in range(2)]
        # Favor the fighter who dealt more damage; fouls cost points
        eff = [dmg_delta[a] - self.foul_points[a] * 0.25 for a in range(2)]
        delta = eff[0] - eff[1]
        if abs(delta) < DRAW_THRESHOLD:
            # Draw round — neither fighter clearly won
            self.round_scores[0].append(10); self.round_scores[1].append(10)
            self.scores[0] += 10; self.scores[1] += 10
        elif delta > 0:
            # Dominant round (big damage gap) = 10-8
            if delta > 15:
                self.round_scores[0].append(10); self.round_scores[1].append(8)
                self.scores[0] += 10; self.scores[1] += 8
            else:
                self.round_scores[0].append(10); self.round_scores[1].append(9)
                self.scores[0] += 10; self.scores[1] += 9
        else:
            if delta < -15:
                self.round_scores[1].append(10); self.round_scores[0].append(8)
                self.scores[1] += 10; self.scores[0] += 8
            else:
                self.round_scores[1].append(10); self.round_scores[0].append(9)
                self.scores[1] += 10; self.scores[0] += 9

    def _decide_decision(self):
        if self.winner is not None:
            return
        if self.scores[0] > self.scores[1]:
            self.winner = 0
        elif self.scores[1] > self.scores[0]:
            self.winner = 1
        else:
            # Scorecards tied — tiebreak by total damage dealt
            dmg = [100 - self.env.hp[a] for a in range(2)]
            dmg_gap = dmg[0] - dmg[1]
            if abs(dmg_gap) < 1.0:
                # Genuine draw — damage within 1 HP. No coin flip.
                self.winner = None  # draw
            else:
                self.winner = 0 if dmg_gap > 0 else 1

    def card(self):
        method = "DECISION"
        if self.ko:
            method = "KO"
        elif self.dq is not None:
            method = "DQ"
        elif getattr(self, "_fell", False):
            method = "FALL"
        elif self.winner is None:
            method = "DRAW"
        return {
            "rounds": self.rounds,
            "round_scores": self.round_scores,
            "total_points": self.scores,
            "foul_points": self.foul_points,
            "winner": self.winner,
            "method": method,
            "final_hp": list(self.env.hp),
        }


def run_bout(env_factory, red_path, blue_path, rounds=ROUNDS,
             round_seconds=ROUND_SECONDS, render=False):
    """Run a full bout under combat rules. red_path/blue_path are policy paths.

    Returns a standardized result dict used by league.py challenge/gauntlet.
    """
    from stable_baselines3 import PPO
    env = env_factory()
    judge = CombatJudge(env, round_seconds=round_seconds, rounds=rounds)
    red = PPO.load(red_path, env=env)
    # blue policy is loaded inside env via opponent_model2 (bout mode)
    obs, _ = env.reset()
    done = False
    while not done:
        # env is single-agent view: agent 0 = red (trained), agent 1 = blue
        # (frozen opponent_model2). Pass red's action; env computes blue.
        a0 = red.predict(obs, deterministic=True)[0]
        actions = a0
        obs, rew, term, trunc, info = judge.step(actions)
        done = term or trunc
    card = judge.card()
    red_win = (card["winner"] == 0)
    return {
        "red_wins": 1 if red_win else 0,
        "blue_wins": 0 if red_win else 1,
        "draws": 0,
        "method": card["method"],
        "red_points": card["total_points"][0],
        "blue_points": card["total_points"][1],
        "fouls": card["foul_points"],
        "final_hp": card["final_hp"],
        "card": card,
    }


# ===========================================================================
# bout_fighter.py — ShadowBoxer + 2-bot bout renderer
# ===========================================================================
#
# 2-bot Track B bout: render a fight between two fighter policies.
#
# Red (r1) vs Blue (r2). Reuses G1FighterEnv (all damage/facing/
# contact logic) with p1 as the trained fighter (r1) and p2 as opponent
# (r2). The env loads the frozen balance policy itself via
# `balance_path` (the substrate); r2 is driven by `opponent_path`
# (r2's own fighter policy) OR a frozen StandPD sandbag.
#
# Outputs an MP4 + prints HP + a scored BoutCard.
#
# Camera is TUNABLE from the CLI so the ropes never sit between
# the lens and the bots (the default is an elevated 3/4 ring view):
#   --cam_az  (deg, 0 = +X / along ring axis; -135 = diagonal)
#   --cam_el  (deg, +up)
#   --cam_dist (m)
#   --cam_lookat "x y z" (defaults to ring center, chest height)
#
# Usage:
#   # real run (trained fighters):
#   python3 combat.py --p1 models/fighter_v1 \
#        --balance models/balance_v1 [--p2 models/fighter_v1] \
#        --out docs/fighter_bout.mp4 --steps 1500
#   # DEMO (no trained fighter yet): scripted shadowboxers so the
#   # bots actually punch + footwork. Uses the balance substrate to
#   # keep them standing. --cam_* to tune the shot.
#   python3 combat.py --balance /tmp/bal_test \
#        --demo --out /tmp/demo_bout.mp4 --steps 900


class ShadowBoxer:
    """Scripted combatant: punches, kicks, dodges, guards, footwork.
    Drives arm joints (14) + walk cmd (3) to create a realistic opponent.

    action (17) = [arm residual 14 | walk cmd 3]
      arm residual 14 -> HOME[15:29] (shoulders/elbows/wrists)
      walk cmd 3        -> (vx, vy, wz) * [0.5, 0.3, 1.0]
    """
    def __init__(self, env, style="red", profile="balanced"):
        self.env = env
        self.style = style
        self.profile = profile
        self.lead = 0 if style == "blue" else 1
        self.t = 0.0
        self.phase = 0.0 if style == "red" else math.pi
        # profile tuning
        if profile == "jabbler":       # fast jabs, aggressive, no defense
            self.cadence = 3.2; self.walk_fwd = 0.6; self.punch_amp = 0.9
            self.dodge_freq = 0.3; self.guard_freq = 0.2; self.kick_freq = 0.1
        elif profile == "defender":   # guard-heavy, counter-puncher, evasive
            self.cadence = 1.6; self.walk_fwd = 0.15; self.punch_amp = 0.6
            self.dodge_freq = 0.8; self.guard_freq = 0.7; self.kick_freq = 0.05
        elif profile == "balanced":   # all-round
            self.cadence = 2.4; self.walk_fwd = 0.4; self.punch_amp = 0.8
            self.dodge_freq = 0.5; self.guard_freq = 0.4; self.kick_freq = 0.15
        else:  # "pd" or default = passive
            self.cadence = 1.0; self.walk_fwd = 0.0; self.punch_amp = 0.3
            self.dodge_freq = 0.1; self.guard_freq = 0.1; self.kick_freq = 0.0

    def predict(self, obs, deterministic=True):
        self.t += 1
        dt = self.env.model.opt.timestep * self.env.frame_skip
        self.phase += dt * self.cadence
        p = self.phase if self.style == "red" else self.phase + math.pi

        arm = np.zeros(14)
        # GUARD position: hands up, elbows bent (default stance)
        arm[0] = -0.7;  arm[3] = 1.3      # L guard
        arm[7] = -0.7;  arm[10] = 1.3     # R guard

        # PUNCH: lead arm extends on the beat
        atk = max(0.0, math.sin(p)) ** 2
        amp = 0.3 * self.punch_amp
        if self.lead == 1:
            arm[7] = -0.7 - amp * atk
            arm[10] = 1.3 - 1.2 * atk
        else:
            arm[0] = -0.7 - amp * atk
            arm[3] = 1.3 - 1.2 * atk
        # COUNTER from rear arm on off-beat
        rear = max(0.0, math.sin(p + math.pi)) ** 2
        if self.lead == 1:
            arm[0] = -0.7 - amp * rear
            arm[3] = 1.3 - 1.1 * rear
        else:
            arm[7] = -0.7 - amp * rear
            arm[10] = 1.3 - 1.1 * rear

        # KICK: occasional leg strike (use walk cmd to simulate)
        # Kicks are signaled by a sharp forward lunge
        kick = 0.0
        if self.kick_freq > 0:
            kick = max(0.0, math.sin(p * 0.5)) ** 2 * self.kick_freq

        # DODGE: lateral movement to evade (weaves side to side)
        dodge = math.sin(p * self.dodge_freq * 2.0) * 0.5

        # GUARD recovery: when not punching, hands snap back to guard
        guard_pulse = max(0.0, math.sin(p + math.pi/2)) * self.guard_freq
        if atk < 0.3:  # not punching hard -> guard up
            arm[0] = -0.7 - 0.2 * guard_pulse
            arm[7] = -0.7 - 0.2 * guard_pulse

        # Footwork: forward pressure + dodge weave + occasional kick lunge
        walk = np.array([
            self.walk_fwd + 0.2 * math.sin(p * 0.5) + 0.5 * kick,  # vx (forward + kick lunge)
            dodge,                                                       # vy (dodge)
            0.4 * math.sin(p * 0.4)                                      # wz (pivot)
        ])
        act = np.concatenate([np.clip(arm, -1, 1), walk]).astype(np.float64)
        return act, None


def _make_camera(args, env):
    """Use the built-in broadcast tracking camera from the model.
    Falls back to free camera if not found."""
    import mujoco
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
    if cam_id >= 0:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id
        return cam
    # fallback: free camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = args.cam_az
    cam.elevation = args.cam_el
    cam.distance = args.cam_dist
    cam.lookat[:] = [float(x) for x in args.cam_lookat.split()]
    return cam


def _bout_main():
    """CLI entry: render a 2-bot bout (was bout_fighter.py main)."""
    import argparse
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    from stable_baselines3 import PPO
    from g1_fighter_env import G1FighterEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default=None,
                    help="fighter policy for r1 (None = shadowboxer demo)")
    ap.add_argument("--p2", default=None,
                    help="fighter policy for r2 (None = shadowboxer demo)")
    ap.add_argument("--balance", default=None,
                    help="balance (substrate) policy path (None = PD stand)")
    ap.add_argument("--out", default="docs/fighter_bout.mp4")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--max_round_seconds", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--demo", action="store_true",
                    help="use scripted ShadowBoxers for both bots")
    ap.add_argument("--no-terminate", action="store_true",
                    help="demo: don't stop on fall/KO (full-length clip)")
    # --- tunable camera (full-body broadcast close-up default) ---
    ap.add_argument("--cam_az", type=float, default=90.0,
                    help="azimuth deg (90 = side-on, bots in profile)")
    ap.add_argument("--cam_el", type=float, default=10.0,
                    help="elevation deg (+up)")
    ap.add_argument("--cam_dist", type=float, default=3.0,
                    help="camera distance (m) -- 3.0 = full bodies in frame")
    ap.add_argument("--cam_lookat", default="-0.15 0 0.7",
                    help="lookat 'x y z' (ring center, mid-body)")
    a = ap.parse_args()

    env = G1FighterEnv(balance_path=a.balance, opponent_path=a.p2,
                       max_steps=a.steps, randomize=False, demo=a.demo)
    judge = CombatJudge(env, round_seconds=a.max_round_seconds,
                        rounds=a.rounds)

    if a.demo or not a.p1:
        p1 = ShadowBoxer(env, style="red")
    else:
        p1 = PPO.load(a.p1)
    # r2: demo -> shadowboxer(blue); else opponent_path (env drives it)
    if a.demo or (a.p2 is None and not a.p1):
        # when demo, drive r2 via opponent_path hook using a ShadowBoxer
        env.opponent = ShadowBoxer(env, style="blue")

    cam = _make_camera(a, env)
    rend = mujoco.Renderer(env.model, height=720, width=1280)

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < a.steps:
        a1, _ = p1.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)
        try:
            rend.update_scene(env.data, camera=cam)
            frames.append(rend.render())
        except Exception:
            pass  # skip frames with EGL errors
        done = (not a.no_terminate) and (
            term or trunc or judge.ko or (judge.winner is not None))
        t += 1

    if frames:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        # imageio (mp4 via ffmpeg) finalizes containers reliably;
        # cv2 mp4v occasionally drops the moov atom on early stop.
        import imageio.v2 as imageio
        # Suppress EGL cleanup errors (known MuJoCo EGL teardown bug)
        import logging
        logging.getLogger('OpenGL').setLevel(logging.CRITICAL)
        try:
            imageio.imsave(a.out, [f[..., ::-1] for f in frames], fps=30, quality=8)
        except Exception:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(a.out, fourcc, 30, (1280, 720))
            for f in frames:
                vw.write(np.ascontiguousarray(f[..., ::-1]))
            vw.release()
        print(f"[saved] {a.out} ({len(frames)} frames)")

    card = judge.card()
    print(f"[bout] CARD: winner={card['winner']} method={card['method']} "
          f"hp={card['final_hp']} rounds={card['round_scores']}")


if __name__ == "__main__":
    # When run directly, behave like the old bout_fighter.py CLI.
    _bout_main()
