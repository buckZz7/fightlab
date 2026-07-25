#!/bin/bash
# GPU pod setup script for FightLab self-play training
# Runs on the RunPod 3090 pod
set -e

echo "=== GPU Pod Setup ==="
nvidia-smi -L
echo ""

# Install dependencies
pip install -q mujoco gymnasium stable-baselines3 onnxruntime imageio joblib

# Clone the repo
cd /workspace
if [ -d fightlab ]; then
    echo "Repo exists, pulling..."
    cd fightlab && git pull
else
    git clone https://github.com/buckZz7/fightlab.git
    cd fightlab
fi

# Copy the ONNX balance policy (not in git, too large)
mkdir -p /workspace/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/
echo "Need to copy policy.onnx to the pod"

# Verify
python -c "
import mujoco, stable_baselines3, onnxruntime
print(f'mujoco: {mujoco.__version__}')
print(f'sb3: {stable_baselines3.__version__}')
print(f'onnxruntime: {onnxruntime.__version__}')
print('GPU ready!')
"

echo ""
echo "=== Setup complete. To train: ==="
echo "cd /workspace/fightlab"
echo "python train_g1_selfplay.py --mocap --timesteps 1000000 --out models/boxing_gen1 --envs 8 --max-steps 2000"
