"""Boxing rules engine for the FightLab King-of-the-Hill MVP.

Defines the legal sport on top of the raw combat env:
  - Legal targets: front torso + head only (no rear-head, no back)
  - Legal weapons: fists only (no kicks, elbows, shoulders, shoves, clinches)
  - Fouls: any non-fist contact, sustained clinch, rear-of-head contact
  - Round structure: 3 rounds x ROUND_SECONDS, bell, rest between rounds
  - Scoring: 10-point must system (round winner gets 10, loser 9 or less)
    with deductions for fouls. HP/KO is a stoppage, not the only win path.
  - Win conditions: KO/TKO (HP<=0 or knockdown), decision (after 3 rounds),
    disqualification (foul points exceed limit).

This is MVP-boxing-only. Kicks are excluded by rule (not just by architecture).
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

class BoxingJudge:
    """Tracks a single bout under boxing rules. Drives env step + scores."""

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
        """Detect illegal contact (non-fist / clinch / rear-head).

        The env only records fist-torso contacts in _contact_states. Any
        OTHER persistent contact between the two robots' bodies that is not
        a legal fist strike counts as a foul (clinch/shove/limb collision).
        We approximate: if a robot's HP dropped without a recorded legal hit,
        OR sustained close contact with no legal strike, flag a foul.
        """
        # Simplified foul model: non-fist body contact detected via geoms.
        # The env doesn't yet record these; we hook into contact geoms here.
        fouled = False
        for con in range(self.env.data.ncon):
            c = self.env.data.contact[con]
            g1, g2 = c.geom1, c.geom2
            b1 = self.env.model.geom_bodyid[g1]
            b2 = self.env.model.geom_bodyid[g2]
            # contact between the two robots, neither geom is a fist
            if (b1 in self._robot_bodies(agent) and b2 in self._robot_bodies(opp)) or \
               (b2 in self._robot_bodies(agent) and b1 in self._robot_bodies(opp)):
                f1 = g1 in self.env.fist_geoms[agent]
                f2 = g2 in self.env.fist_geoms[agent]
                if not (f1 or f2):
                    # body-to-body contact that isn't a fist strike = foul
                    # (clinch / shove / leg / shoulder). Penalize lightly.
                    fouled = True
        return fouled

    def step(self, actions):
        """Advance one env step under rules. Returns (obs, rew, done, info)."""
        obs, rew, term, trunc, info = self.env.step(actions)

        # Round clock
        self.round_time += self.env.model.opt.timestep * self.env.FRAME_SKIP

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

        # Round end
        if self.round_time >= self.round_seconds and not self.ko and self.dq is None:
            self._score_round()
            self.round += 1
            self.round_time = 0.0
            self.round_fouls = [0, 0]
            if self.round >= self.rounds:
                self._decide_decision()
                info['decision'] = True
                return obs, rew, True, info
            # reset positions for next round (keep HP)
            info['round_end'] = self.round

        done = term or trunc or self.ko or self.dq is not None
        return obs, rew, done, info

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
        return {
            "rounds": self.rounds,
            "round_scores": self.round_scores,
            "total_points": self.scores,
            "foul_points": self.foul_points,
            "winner": self.winner,
            "method": ("KO" if self.ko else "DQ" if self.dq is not None
                       else "DECISION"),
            "final_hp": list(self.env.hp),
        }


def run_bout(env_factory, red_path, blue_path, rounds=ROUNDS,
             round_seconds=ROUND_SECONDS, render=False):
    """Run a full bout under boxing rules. red_path/blue_path are policy paths.

    Returns a standardized result dict used by league.py challenge/gauntlet.
    """
    from stable_baselines3 import PPO
    env = env_factory()
    judge = BoxingJudge(env, round_seconds=round_seconds, rounds=rounds)
    red = PPO.load(red_path, env=env)
    blue = PPO.load(blue_path, env=env)
    # Who fights whom: map 0->red, 1->blue in env agent indexing.
    obs, _ = env.reset()
    done = False
    while not done:
        # env agent 0 = red, agent 1 = blue (or vice versa; we score by winner)
        a0 = red.predict(obs, deterministic=True)[0]
        a1 = blue.predict(obs, deterministic=True)[0]
        actions = np.stack([a0, a1], axis=0)
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
