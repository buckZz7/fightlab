#!/bin/bash
# Full pipeline: tracker -> combat -> league -> title bout
# Run on the EGL pod after tracker training completes.

set -e
cd /workspace
export G1_SCENE_XML=/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
export G1_MESH_DIR=/workspace/unitree_mujoco/unitree_robots/g1/meshes
export MUJOCO_GL=egl

TRACKER=models/motion_tracker_v2.zip
FIGHTER=models/fighter_v2.zip

echo "=== Stage 2: Combat Fine-Tuning ==="
if [ ! -f "$TRACKER" ]; then
    echo "ERROR: tracker not found at $TRACKER"
    exit 1
fi

python3 train_combat.py --tracker $TRACKER --steps 1000000 --envs 16 --out models/fighter_v2

echo "=== Stage 3: League Entry ==="
python3 league_update.py --standings docs/league_standings.json --bouts 3 --max_steps 5000 --render-steps 5000 --max-render-bouts 4

echo "=== Stage 4: Title Bout Render ==="
# Find the king from standings
KING=$(python3 -c "import json; d=json.load(open('docs/league_standings.json')); print(d.get('king','unknown'))")
echo "King: $KING"

# Render the title bout (fighter_v2 vs king)
python3 egl_bout.py --p1 models/fighter_v2.zip --steps 5000 --out /tmp/title_bout.mp4 --no-terminate || true

echo "=== DONE ==="
echo "King: $KING"
echo "Title bout: /tmp/title_bout.mp4"
