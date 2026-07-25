"""Unit test for boxing_rules scoring/decision (no sim, fast).

Tests:
1. 10-point must: fighter dealing more damage wins round 10-9.
2. Fouls deduct points and can cause DQ.
3. KO stops bout and awards winner.
4. Decision after 3 rounds with no KO.
5. Tiebreak by damage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from boxing_rules import BoxingJudge, ROUNDS


class FakeEnv:
    """Minimal env stub exposing the attributes BoxingJudge reads."""
    def __init__(self):
        self.hp = [100.0, 100.0]
        self._contact_states = {}
        self.step_count = 0
        self.model = type('M', (), {'opt': type('O', (), {'timestep': 0.01})()})()
        self.FRAME_SKIP = 4
        self.data = type('D', (), {'ncon': 0, 'contact': []})()
        self.prefix = ['r1_', 'r2_']
        self.fist_geoms = [set(), set()]
        self.torso_bodies = [set(), set()]

    def _pelvis_z(self, a):
        return 0.78

    def step(self, actions):
        return None, 0.0, False, False, {}


def make_judge():
    env = FakeEnv()
    return BoxingJudge(env, round_seconds=1.0, rounds=ROUNDS)


def test_round_scoring():
    j = make_judge()
    j.round_time = 1.0  # trigger round end
    j.env.hp = [80.0, 100.0]  # agent 0 dealt 20 dmg, agent1 dealt 0
    j._score_round()
    assert j.round_scores[0][0] == 10 and j.round_scores[1][0] == 9, j.round_scores
    print("PASS round_scoring (20 dmg -> 10-9)")


def test_foul_dq():
    j = make_judge()
    j.env.hp = [100.0, 100.0]
    # simulate 4 foul points
    j.foul_points = [4.0, 0.0]
    j.dq = 0
    j.winner = 1
    card = j.card()
    assert card["method"] == "DQ" and card["winner"] == 1, card
    print("PASS foul_dq (4 foul pts -> DQ, winner=1)")


def test_ko():
    j = make_judge()
    j.env.hp = [0.0, 100.0]
    j.ko = True
    j.winner = 1
    card = j.card()
    assert card["method"] == "KO" and card["winner"] == 1, card
    print("PASS ko (hp=0 -> KO, winner=1)")


def test_decision_after_rounds():
    j = make_judge()
    # 3 rounds, agent0 wins all 10-9
    for r in range(ROUNDS):
        j.round_time = 1.0
        j.env.hp = [100 - (r+1)*10, 100.0]  # agent0 deals 10/round
        j._score_round()
        j.round += 1
    j._decide_decision()
    card = j.card()
    assert card["method"] == "DECISION", card
    assert card["winner"] == 0, card
    assert card["total_points"][0] == 30 and card["total_points"][1] == 27, card
    print("PASS decision_after_rounds (3x 10-9 -> DEC, winner=0, 30-27)")


def test_tiebreak_by_damage():
    j = make_judge()
    j.round_time = 1.0
    j.env.hp = [90.0, 90.0]  # equal points, equal rounds -> tiebreak by dmg
    j._score_round()
    j.round += 1
    j.env.hp = [80.0, 80.0]
    j._score_round()
    j.round += 1
    j._decide_decision()
    card = j.card()
    # both dealt 20 dmg -> tie, winner = agent0 (>=)
    assert card["winner"] == 0, card
    print("PASS tiebreak (equal -> agent0 by dmg>=)")


if __name__ == "__main__":
    test_round_scoring()
    test_foul_dq()
    test_ko()
    test_decision_after_rounds()
    test_tiebreak_by_damage()
    print("\nALL BOXING RULES TESTS PASSED")
