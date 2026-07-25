#!/usr/bin/env bash
# RunPod pod bootstrap: install deps, run mocap punch training, upload result.
set -ex
cd /workspace

# system python deps (pod has GPU torch preinstalled via runpod/pytorch image)
pip install --no-cache-dir mujoco gymnasium stable-baselines3 onnxruntime joblib imageio imageio-ffmpeg matplotlib huggingface_hub

# repo (public)
git clone --depth 1 https://github.com/buckZz7/fightlab.git repo
cd repo

# frozen whole-body policy is committed? no — it's in unitree_rl_mjlab. Fetch it.
mkdir -p /workspace/rl_mjlab
cd /workspace/rl_mjlab
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_mjlab.git .
git sparse-checkout set deploy/robots/g1/config/policy/velocity/v0 src/assets/robots/unitree_g1
cd /workspace/repo

# point loco_base29 at the pod path if the default path is missing
if [ ! -f /opt/data/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx ]; then
  mkdir -p /opt/data/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported
  cp /workspace/rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
     /opt/data/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/
fi

# G1 scene XML (unitree_mujoco)
mkdir -p /opt/data
cd /opt/data
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_mujoco.git .
git sparse-checkout set unitree_robots/g1

cd /workspace/repo
python train_g1_mocap_punch.py --timesteps ${TIMESTEPS:-800000} --out /workspace/g1_mocap_punch --envs ${ENVS:-16}

# render 3D footage of the trained policy on the GPU (EGL hardware GL)
MUJOCO_GL=egl python g1_render_3d.py --model /workspace/g1_mocap_punch.zip --out /workspace/punch_3d.mp4 --seconds 12 || echo "render failed, continuing"

# upload result back: tar the model + video + logs
tar czf /workspace/g1_mocap_punch_result.tar.gz -C /workspace g1_mocap_punch.zip g1_mocap_punch_ckpt punch_3d.mp4 2>/dev/null || true
echo "DONE"
