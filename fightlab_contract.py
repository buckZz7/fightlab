"""FightLab miner contract -- the OFFICIAL obs/act interface.

A CHALLENGER is a policy that maps OBS (85,) -> ACT (17,).
Miners train their own policy (any framework: SB3, MJX, Isaac
Lab, raw PyTorch) but MUST match this exact interface to be
evaluated in the FightLab arena.

Design: the core provides a FROZEN balance substrate
(models/balance_v1). Every challenger stands on the SAME legs,
so miners compete on STRIKING + (future) FOOTWORK strategy,
not on re-solving locomotion. This is the fairness boundary.

Eval: a deterministic seeded bout vs the reigning KING.
Scoring: boxing_rules.BoxingJudge (10-point must, damage, fouls).

Version: v1 -- frozen balance; arms + walk-cmd control.
"""
import numpy as np

OBS_DIM = 85
ACT_DIM = 17

# --- OBS layout (float64, contiguous, length 85) ---
# field : (start, end)  source in g1_fighter_env._get_obs(agent)
OBS_LAYOUT = {
    "quat":      (0, 4),    # torso orientation (x,y,z,w)  qpos[off+3:off+7]
    "angvel":    (4, 7),    # torso angular velocity (rad/s)
    "jrel":      (7, 36),   # 29 joint angles minus HOME (stand pose)
    "jvel":      (36, 65),  # 29 joint velocities (rad/s)
    "hp_self":   (65, 66),  # own HP (0..100)
    "hp_opp":    (66, 67),  # opponent HP (0..100)
    "rel":       (67, 70),  # opponent pelvis - own pelvis (3,)
    "pelvis_z":  (70, 71),  # own pelvis height (m)
    "residuals": (71, 85),  # 14 smoothed arm-action residuals (env feedback)
}

# --- ACT layout (float64, contiguous, length 17) ---
# field : (start, end)  interpretation
ACT_LAYOUT = {
    "arm_residual": (0, 14),   # 14 arm-joint offsets, in [-1,1].
                              #   env: smoothed (lerp 0.25) * RESIDUAL_SCALE(0.15)
                              #   + added to HOME[15:29] (arm joints 15..28).
    "walk_cmd":     (14, 17),  # (vx, vy, wz) in [-1,1], scaled by (0.5, 0.3, 1.0).
                              #   v1 LIMITATION: footwork integration pending
                              #   (frozen balance base does not yet consume it).
                              #   Accepted for forward-compat; see V1_LIMITS.
}

RESIDUAL_SCALE = 0.15
WALK_SCALE = np.array([0.5, 0.3, 1.0])

V1_LIMITS = [
    "Footwork: walk_cmd is accepted but the frozen balance base does not "
    "yet consume it, so v1 challengers are STANDING PUNCHERS (stationary). "
    "Real approach/footwork arrives in v2 with a walkable balance base.",
    "Balance is provided (frozen). Miners do NOT train or submit legs; "
    "they submit arm+walk strategy only. This keeps bouts fair.",
    "Single observation per step (no history/lstm). Use the residuals "
    "field as short-term action feedback.",
]

DETERMINISM = [
    "Bouts are seeded: env reset uses a fixed seed per round.",
    "The KING model is fixed for a title reign; challengers are matched "
    "against the same KING for fair ranking.",
    "Scoring is via boxing_rules.BoxingJudge (no randomness in judging).",
]


class Challenger:
    """Miners implement this interface (or submit a compatible predict())."""

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return ACT (17,) given OBS (85,). Must be deterministic."""
        raise NotImplementedError

    def reset(self):
        """Called at the start of each bout (optional)."""
        pass


def validate_policy(pol, obs_dim=OBS_DIM, act_dim=ACT_DIM):
    """Check a policy conforms: predict(obs85)->act17, clipped, deterministic."""
    o = np.zeros(obs_dim, dtype=np.float64)
    a1 = np.asarray(pol.predict(o), dtype=np.float64)
    a2 = np.asarray(pol.predict(o), dtype=np.float64)
    assert a1.shape == (act_dim,), f"act shape {a1.shape} != ({act_dim},)"
    assert np.allclose(a1, a2), "policy is non-deterministic"
    assert np.all(np.abs(a1) <= 1.0 + 1e-6), "act outside [-1,1]"
    return True


def obs_slice(obs, field):
    s, e = OBS_LAYOUT[field]
    return obs[s:e]


def act_slice(act, field):
    s, e = ACT_LAYOUT[field]
    return act[s:e]
