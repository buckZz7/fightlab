"""Post-train pipeline for Track B: when fighter_v1.zip appears, validate
it, crown it as the Track B genesis king (replacing the old frozen-base
MVP king), render the inaugural title bout, and stage assets for the
site.

Pure plumbing -- does not depend on the balance model at runtime
(just needs models/balance_v1.zip present for the bout env, and
fighter_v1.zip to exist). Designed to run on the pod after
train_fighter.py finishes.

Usage (pod):
  python post_fighter.py            # polls for models/fighter_v1.zip
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(__file__))

FIGHTER = "models/fighter_v1.zip"
BALANCE = "models/balance_v1.zip"
OLD_KING = "models/boxing_gen2.zip"   # previous MVP baseline, for the showcase

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
    print("Waiting for", FIGHTER, "...")
    while not os.path.exists(FIGHTER):
        time.sleep(30)
    print(f"Found {FIGHTER}. Running Track B post-train pipeline.")

    # 0. sanity: balance substrate present
    if not os.path.exists(BALANCE):
        print("[ABORT] balance substrate missing:", BALANCE)
        sys.exit(1)

    # 1. self-test: does the fighter stand + land hits? (eval_fighter.py)
    run(f"python eval_fighter.py --policy {FIGHTER} --balance {BALANCE} --episodes 5")

    # 2. crown as Track B genesis king (force: replaces old frozen-base king)
    run(f"python league.py crown {FIGHTER} "
        f"--cause 'Track B genesis: balance_v1 + 2M fighter (full-body RL)' --force")

    # 3. render inaugural title bout: new king (red/p1) vs old MVP king (blue/p2)
    if os.path.exists(OLD_KING):
        opp = OLD_KING
        label = "trackB_vs_mvp"
    else:
        opp = FIGHTER   # fallback: self-bout demo
        label = "trackB_demo"
    run(f"python bout_fighter.py --p1 {FIGHTER} --p2 {opp} "
        f"--balance {BALANCE} --out title_bout.mp4 --steps 1500")

    # 4. export kings.jsonl -> docs/kings.json for the website
    run("python -c \"import json; "
        "ks=[json.loads(l) for l in open('models/kings.jsonl') if l.strip()]; "
        "json.dump(ks, open('docs/kings.json','w'), indent=2)\"")

    # 5. stage assets into docs/ for GitHub Pages
    run("cp title_bout.mp4 docs/title_bout.mp4")
    run("cp models/kings.jsonl docs/kings.json")
    run("python -c \"import json; "
        "d=json.load(open('docs/kings.json')); "
        "print('kings:', [(k['gen'], k['path'], k['elo']) for k in d])\"")

    # 6. status
    run("python league.py status")
    print("DONE. title_bout.mp4 + kings.json in docs/ ready to commit + push.")
    print("Next: git add docs/ && git commit -m 'Track B king' && git push")


if __name__ == "__main__":
    main()
