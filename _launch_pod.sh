#!/bin/bash
# Launch balance training + fighter watcher, fully detached.
export G1_SCENE_XML=/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
export G1_MESH_DIR=/workspace/unitree_mujoco/unitree_robots/g1/meshes
export MUJOCO_GL=osmesa
cd /workspace/repo
nohup python3 train_balance.py --out models/balance_v1 --n_envs 16 > balance_train.log 2>&1 &
echo "balance pid $!"
nohup python3 watch_and_launch_fighter.py > watch.log 2>&1 &
echo "watch pid $!"
