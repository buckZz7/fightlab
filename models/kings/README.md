# Kings Archive

This directory contains the weights of all past kings. Anyone can download
these and use them as:
1. A starting point for fine-tuning (load the king's weights, improve on top)
2. A training opponent (train your fighter to beat the king's strategy)

## Current King
See docs/league_standings.json for the current champion.

## How to use a king's weights

### As a fine-tuning starting point:
```python
from stable_baselines3 import PPO
model = PPO.load('models/kings/fighter_walker_v1.zip', env=env)
model.learn(total_timesteps=500000)  # fine-tune
```

### As a training opponent:
```bash
python3 train_fight.py --opponent models/kings/fighter_walker_v1.zip --out models/my_fighter
```

### To challenge the king directly:
```bash
python3 deterministic_eval.py --fighter models/my_fighter.zip --king models/kings/fighter_walker_v1.zip
```

## Rules
- Kings' weights are open. Anyone can download, study, and build on them.
- To take the crown, you must BEAT the king (not copy it). The CI gate
  requires winning on 2+ seeds with actual damage dealt.
- A draw means the king stays. You have to win.
