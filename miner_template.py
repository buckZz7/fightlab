"""MINER TEMPLATE -- copy this to implement a FightLab challenger.

A challenger maps OBS (85,) -> ACT (17,) and is submitted as an SB3
.zip (or any object exposing .predict(obs, deterministic=True)
-> (act17, None)). The core runs it through league.run_fighter_bout
against the reigning king; win >=60% of a 15-bout series to dethrone.

This stub stands still (safe baseline). Replace predict() with your
trained policy. See MINER_SPEC.md for the full obs/act layout.

Contract (see fightlab_contract.py):
  OBS (85): quat4 | angvel3 | jrel29 | jvel29 | hp_self1 |
            hp_opp1 | rel3 | pelvis_z1 | residuals14
  ACT (17): arm_residual14 | walk_cmd3   (each in [-1, 1])
"""
import numpy as np


class Challenger:
    ACT_DIM = 17

    def __init__(self, weights_path=None):
        # TODO: load your trained net here
        self.act_dim = self.ACT_DIM

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """obs: (85,) float -> act: (17,) in [-1, 1]."""
        # Placeholder: stand still. A real policy infers from obs.
        return np.zeros(self.act_dim, dtype=np.float64), None


# ---------------------------------------------------------------------------
# Local self-test: confirm the contract shapes are satisfiable.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    c = Challenger()
    obs = np.zeros(85)
    act, _ = c.predict(obs)
    assert act.shape == (17,), act.shape
    assert act.min() >= -1.0 and act.max() <= 1.0
    print("[ok] Challenger contract: obs(85) -> act(17) satisfied")
