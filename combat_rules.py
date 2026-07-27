"""Combat rules engine for the FightLab King-of-the-Hill league.

Defines the legal sport on top of the raw combat env:
  - Legal targets: torso + head (front and side)
  - Legal weapons: fists, feet, knees, elbows (full combat)
  - Round structure: 3 rounds x ROUND_SECONDS, bell, rest between rounds
  - Scoring: 10-point must system (round winner gets 10, loser 9 or less)
    with deductions for fouls. HP/KO is a stoppage, not the only win path.
  - Win conditions: KO/TKO (HP<=0 or knockdown), decision (after 3 rounds).

Full combat: punches, kicks, spins all count. Any clean hit with
sufficient relative velocity = damage.
"""
import json
import numpy as np
from datetime import datetime, timezone

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
        """10-point must system: round winner gets 10, loser 9 (or less)."""
        # Compare legal damage dealt this round
        dmg = [100 - self.env.hp[a] for a in range(2)]
        dmg_delta = [dmg[a] - dmg[1 - a] for a in range(2)]
        # Favor the fighter who dealt more damage; fouls cost points
        eff = [dmg_delta[a] - self.foul_points[a] * 0.25 for a in range(2)]
        if eff[0] > eff[1]:
            self.round_scores[0].append(10); self.round_scores[1].append(9)
            self.scores[0] += 10; self.scores[1] += 9
        elif eff[1] > eff[0]:
            self.round_scores[1].append(10); self.round_scores[0].append(9)
            self.scores[1] += 10; self.scores[0] += 9
        else:
            self.round_scores[0].append(10); self.round_scores[1].append(10)
            self.scores[0] += 10; self.scores[1] += 10

    def _decide_decision(self):
        if self.winner is not None:
            return
        if self.scores[0] > self.scores[1]:
            self.winner = 0
        elif self.scores[1] > self.scores[0]:
            self.winner = 1
        else:
            # tiebreak by total damage dealt
            dmg = [100 - self.env.hp[a] for a in range(2)]
            self.winner = 0 if dmg[0] >= dmg[1] else 1

    def card(self):
        method = "DECISION"
        if self.ko:
            method = "KO"
        elif self.dq is not None:
            method = "DQ"
        elif getattr(self, "_fell", False):
            method = "FALL"
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
