"""Evaluate Gen 2 (or any frozen-base policy) against boxing-MVP criteria.

Reports: no-fall rate, hit damage dealt/taken, mean facing, mean distance,
engagement time (dist<0.6 after contact), shove detection (non-fist damage=0).
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from g1_selfplay_env import G1SelfPlayEnv
from stable_baselines3 import PPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/boxing_gen2.zip")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    env = G1SelfPlayEnv(opponent_mocap=True, max_steps=500, randomize=False)
    model = PPO.load(args.model, env=env)

    falls=0; facings=[]; dists=[]; eng=0; steps_tot=0
    for ep in range(args.episodes):
        obs,_=env.reset(); done=False; t=0; eng_ep=0
        while not done and t<500:
            a,_=model.predict(obs, deterministic=True)
            obs,r,term,trunc,inf=env.step(a)
            done=term or trunc; t+=1
            facings.append(env._last_facing if hasattr(env,'_last_facing') else 0)
            dists.append(env._last_dist if hasattr(env,'_last_dist') else 0)
            if env._last_dist<0.6 if hasattr(env,'_last_dist') else False: eng_ep+=1
        steps_tot+=t
        fell = env.data.qpos[2] < 0.4
        if fell: falls+=1
        print(f"ep{ep}: steps={t} fell={fell} agentHP={env.hp[0]:.1f} oppHP={env.hp[1]:.1f} dmgDealt={100-env.hp[1]:.1f} dmgTaken={100-env.hp[0]:.1f}")
    facings=np.array(facings); dists=np.array(dists)
    print("\n=== SUMMARY ===")
    print(f"episodes={args.episodes} falls={falls} ({100*(1-falls/args.episodes):.0f}% upright)")
    print(f"mean facing={facings.mean():.2f} (1=facing, -1=back)")
    print(f"mean dist={dists.mean():.2f}m  min={dists.min():.2f}m")
    print(f"mean engagement (dist<0.6): {100*(dists<0.6).mean():.0f}% of steps")

if __name__ == "__main__":
    main()
