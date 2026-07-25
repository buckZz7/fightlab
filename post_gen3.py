"""Post-train pipeline: when Gen 3 model appears, crown it (if it beats Gen 2)
and render the title bout video.

Run on the pod after launching Gen 3 training:
  python post_gen3.py

Polls for models/boxing_gen3.zip; when found, runs:
  1. auto_crown.py models/boxing_gen3.zip  (crown if >=60% vs Gen 2)
  2. render_bout.py models/boxing_gen3.zip models/boxing_gen2.zip --out gen3_title_bout.mp4
Then pulls results back / reports.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import subprocess

MODEL = "models/boxing_gen3.zip"
KING = "models/boxing_gen2.zip"
ENV = dict(os.environ,
           G1_SCENE_XML="/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
           G1_MESH_DIR="/workspace/unitree_mujoco/unitree_robots/g1/meshes",
           G1_ONNX_PATH="/workspace/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx",
           MUJOCO_GL="osmesa")

def run(cmd):
    print(f">>> {cmd}")
    r = subprocess.run(cmd, shell=True, env=ENV, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[-1500:])
    return r.returncode

def main():
    print("Waiting for", MODEL, "...")
    while not os.path.exists(MODEL):
        time.sleep(30)
    print(f"Found {MODEL}. Running post-train pipeline.")
    # 1. Crown if it beats the king
    run(f"python auto_crown.py {MODEL} --matches 5 --rounds 3 --round-seconds 30")
    # 2. Render the title bout (gen3 challenger vs gen2 king)
    run(f"python render_bout.py {MODEL} {KING} --out gen3_title_bout.mp4 "
        f"--rounds 3 --round-seconds 30 --max-steps 2000")
    # 3. Status
    run("python league.py status")
    print("DONE. Pull gen3_title_bout.mp4 and models/kings.jsonl")

if __name__ == "__main__":
    main()
