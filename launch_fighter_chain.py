"""One-shot launch sequencer for the Track B training chain.

Polls for models/balance_v1.zip. When it appears:
  1. Run preflight_fighter.py (frozen-substrate stand check inside
     G1FighterEnv). Aborts if it fails.
  2. Launch train_fighter.py (2M steps, also self-gates) in background.
  3. Poll for models/fighter_v1.zip; when it appears, run
     post_fighter.py (crown + render + stage for site).

Run on the pod (background) after balance training starts:
  python3 launch_fighter_chain.py &
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))

BALANCE = "models/balance_v1.zip"
FIGHTER = "models/fighter_v1.zip"
ENV = dict(os.environ,
           G1_SCENE_XML="/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
           G1_MESH_DIR="/workspace/unitree_mujoco/unitree_robots/g1/meshes",
           MUJOCO_GL="osmesa",
           BALANCE_PATH="/workspace/repo/models/balance_v1")


def run(cmd):
    print(f">>> {cmd}")
    r = subprocess.run(cmd, shell=True, env=ENV, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[-1500:])
    return r.returncode


def main():
    print("Waiting for", BALANCE, "...")
    while not os.path.exists(BALANCE):
        time.sleep(20)
    print(f"[chain] {BALANCE} ready. Running fighter pre-flight.")

    rc = run("python preflight_fighter.py")
    if rc != 0:
        print("[chain] ABORT: fighter substrate failed pre-flight. Fix before training.")
        sys.exit(1)

    print("[chain] Launching train_fighter.py (2M steps)...")
    # launch in background via setsid so it survives; log to fighter_train.log
    subprocess.Popen(
        "setsid python train_fighter.py --steps 2_000_000 --out models/fighter_v1 "
        "> fighter_train.log 2>&1 < /dev/null &",
        shell=True, env=ENV)
    print("[chain] fighter training launched (see fighter_train.log). Polling for", FIGHTER)

    while not os.path.exists(FIGHTER):
        time.sleep(60)
        # still alive?
        alive = subprocess.run("pgrep -f train_fighter.py | head -1",
                               shell=True, env=ENV, capture_output=True, text=True).stdout.strip()
        if not alive:
            print("[chain] train_fighter.py not running but", FIGHTER,
                  "not found -- check fighter_train.log")
            # give it a moment; if really gone and no model, abort
            if not os.path.exists(FIGHTER):
                print("[chain] ABORT: fighter training died without producing model.")
                sys.exit(1)

    print(f"[chain] {FIGHTER} ready. Running post_fighter.py (crown + render + stage).")
    run("python post_fighter.py")
    print("[chain] DONE. Title bout + king card staged in docs/. Commit + push.")


if __name__ == "__main__":
    main()
