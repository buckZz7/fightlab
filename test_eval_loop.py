"""Miner reference implementation + Gittensor eval-loop smoke test.

Proves the submission contract (obs85 -> act17) is implementable and
that league.run_fighter_bout() scores a challenger end-to-end -- using a
MOCK challenger (no trained king needed). This de-risks the whole
subnet eval path BEFORE real policies exist.

Run on pod (needs models/balance_v1.zip for the env substrate):
  python3 test_eval_loop.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np


class MockChallenger:
    """Reference miner policy: implements the obs85->act17 contract.

    A real miner would load their trained net in predict(). This mock
    returns a safe stand-still action (zeros) so we can exercise the
    full eval loop without a trained model.
    """
    act_dim = 17

    def predict(self, obs, deterministic=True):
        # obs is (85,); return (17,) in [-1, 1]
        return np.zeros(self.act_dim, dtype=np.float64), None


def main():
    from league import run_fighter_bout, load_policy

    bal = os.environ.get("BALANCE_PATH", "models/balance_v1")
    if not os.path.exists(bal):
        print(f"[SKIP] balance substrate missing: {bal} (eval loop needs it)")
        sys.exit(0)

    # opponent = any real .zip (used as r2). Fall back to the mock as r2
    # by passing a path that load_policy can open; if none, use mock for both.
    king_path = os.environ.get("KING_PATH", "models/boxing_gen2.zip")
    if not os.path.exists(king_path):
        print(f"[note] no king .zip at {king_path}; using MockChallenger as r2 too")
        # run_fighter_bout wants a king PATH for opponent; build env manually
        from g1_fighter_env import G1FighterEnv
        env = G1FighterEnv(balance_path=bal, opponent_path=None,
                           max_steps=300, randomize=False)
        chall = MockChallenger()
        o, _ = env.reset(seed=0)
        for step in range(300):
            a, _ = chall.predict(o)
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        print(f"[ok] mock bout ran {step+1} steps, hp0={env.hp[0]:.0f} "
              f"hp1={env.hp[1]:.0f}, no crash")
        return

    chall = MockChallenger()
    res = run_fighter_bout(chall, king_path, bal, matches=1, max_steps=300)
    print("[ok] run_fighter_bout returned:", {k: res[k] for k in
          ("red_wins", "blue_wins", "draws")})
    print("sample card:", res["cards"][0])


if __name__ == "__main__":
    main()
