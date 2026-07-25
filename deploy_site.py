"""Deploy the latest bout render + king card to the FightLab GitHub Pages site.

Run after post_gen3.py finishes (gen3_title_bout.mp4 + docs/kings.json exist).
Commits docs/ and pushes -> GitHub Actions rebuilds the site.
"""
import os, subprocess, sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def sh(c):
    print(">>>", c)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("ERR:", r.stderr[-800:])
    return r.returncode

if __name__ == "__main__":
    # sanity: video exists
    vid = "docs/gen3_title_bout.mp4"
    if not os.path.exists(vid):
        print("NO VIDEO at", vid, "- run post_gen3.py first")
        sys.exit(1)
    sz = os.path.getsize(vid)
    print(f"Video {vid}: {sz/1e6:.1f} MB")
    sh("git add docs/gen3_title_bout.mp4 docs/kings.json docs/kings.jsonl")
    sh('git commit -m "deploy: Gen3 title bout render + live king card"')
    sh("git push origin main")
    print("PUSHED. GitHub Pages will rebuild at buckZz7.github.io/fightlab/")
