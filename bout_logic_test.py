"""Headless bout LOGIC test (no rendering) -- proves BoxingJudge + G1FighterEnv
step loop works end-to-end without a trained model.

Run: python3 bout_logic_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from g1_fighter_env import G1FighterEnv
from boxing_rules import BoxingJudge


class RandAct:
    def __init__(self, env): self.env = env
    def predict(self, obs, deterministic=True):
        return self.env.action_space.sample(), None


def main():
    env = G1FighterEnv(balance_path=None, opponent_path=None,
                       max_steps=1500, randomize=False)  # full 3x30s-capable
    judge = BoxingJudge(env, round_seconds=3.0, rounds=3)  # 3s rounds for speed
    red = RandAct(env)
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < 1500:
        a0 = red.predict(obs)[0]
        obs, rew, term, trunc, info = judge.step(a0)
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1
    card = judge.card()
    print(f"[logic] steps={t} rounds_played={len(card['round_scores'][0])}")
    print(f"[logic] CARD: winner={card['winner']} method={card['method']} "
          f"pts={card['total_points']} hp={card['final_hp']} fouls={card['foul_points']}")
    # assertions: pipeline must not crash + produce a valid result
    # (a fall/KO can end the bout before any round is fully scored)
    assert card["winner"] in (0, 1), f"winner must be set: {card['winner']}"
    assert len(card["round_scores"][0]) >= 1 or card["method"] in ("KO", "TKO", "FALL"), \
        "either rounds scored or a stoppage occurred"
    print("[logic] PASS -- bout pipeline runs end-to-end (no trained model)")


if __name__ == "__main__":
    main()
