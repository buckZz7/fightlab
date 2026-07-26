#!/usr/bin/env bash
# One-shot setup for a fresh RunPod fightlab pod.
# Run ON the pod after SSH. Installs deps, clones/scps repo, launches balance->fighter watcher.
set -e
export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL=osmesa
export G1_SCENE_XML=/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
export G1_MESH_DIR=/workspace/unitree_mujoco/unitree_robots/g1/meshes

echo "[setup] installing system deps (OSMesa + pip)"
apt-get update -qq
apt-get install -y -qq build-essential libosmesa6-dev libgl1-mesa-dev wget git >/dev/null 2>&1 || true

echo "[setup] python venv + pkgs"
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy mujoco==3.2.4 gymnasium stable_baselines3 imageio imageio-ffmpeg opencv-python-headless pyyaml

echo "[setup] unitree_mujoco (G1 meshes + scene)"
if [ ! -d /workspace/unitree_mujoco ]; then
  git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco /workspace/unitree_mujoco
fi

echo "[setup] repo already scp'd to /workspace/repo"
cd /workspace/repo
ls -la

echo "[setup] launching balance training (watcher auto-launches fighter)"
export G1_SCENE_XML G1_MESH_DIR MUJOCO_GL
setsid python3 train_balance.py --out models/balance_v1 --n_envs 16 > balance_train.log 2>&1 < /dev/null &
setsid python3 watch_and_launch_fighter.py > watch.log 2>&1 < /dev/null &
echo "[setup] DONE. tail -f balance_train.log"
