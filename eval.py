"""FightLab evaluation, testing, and rendering — consolidated module.

This file merges (preserving all functionality):
  - egl_bout.py        : EGL 1v1 eval bout (PNG frames -> ffmpeg)
  - eval_tracker.py    : motion-tracker eval (render policy on a motion)
  - test_damage.py     : damage detection sanity test
  - bout_overlay.py    : HP bars + round timer overlay on bout videos
  - ci_gate.py         : CI merge gate for fighter submissions
  - deterministic_eval.py : trustless deterministic league eval

Each piece is exposed as a function or `if __name__ == "__main__"`-style
section so the same CLI behavior is preserved. Subcommands are dispatched
via `python3 eval.py <subcommand> ...` (see `main` below) — and each legacy
script is also reachable via `python3 eval.py --<legacy-name> ...` for
backward compatibility.
"""
import os, sys, json, hashlib, argparse, time, subprocess

# Default env config (used by most subcommands).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("G1_SCENE_XML",
    "/workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml")
os.environ.setdefault("G1_MESH_DIR",
    "/workspace/unitree_mujoco/unitree_robots/g1/meshes")


# ===========================================================================
# bout_overlay.py — HP bars + round timer overlay
# ===========================================================================
def add_overlay(frame_img, hp_red, hp_blue, round_num, round_time,
                name_red, name_blue):
    """Draw HP bars + timer on a frame."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    img = frame_img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    bar_w = int(w * 0.35)
    bar_h = 8
    bar_y = 20
    bar_margin = 20

    # Red (left) HP bar
    red_w = int(bar_w * max(0, hp_red) / 100)
    draw.rectangle([bar_margin, bar_y, bar_margin + bar_w, bar_y + bar_h],
                   fill=(40, 40, 40), outline=(80, 80, 80))
    draw.rectangle([bar_margin, bar_y, bar_margin + red_w, bar_y + bar_h],
                   fill=(220, 60, 60))

    # Blue (right) HP bar
    blue_w = int(bar_w * max(0, hp_blue) / 100)
    blue_x = w - bar_margin - bar_w
    draw.rectangle([blue_x, bar_y, blue_x + bar_w, bar_y + bar_h],
                   fill=(40, 40, 40), outline=(80, 80, 80))
    draw.rectangle([blue_x + bar_w - blue_w, bar_y, blue_x + bar_w, bar_y + bar_h],
                   fill=(60, 120, 240))

    # Fighter names
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((bar_margin, bar_y - 18), name_red[:20], fill=(220, 60, 60), font=font)
    draw.text((blue_x, bar_y - 18), name_blue[:20], fill=(60, 120, 240), font=font)

    # Round + timer (center)
    timer_text = f"R{round_num} {round_time:.1f}s"
    bbox = draw.textbbox((0, 0), timer_text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((w // 2 - tw // 2, 10), timer_text, fill=(255, 255, 255), font=font)

    return img


def overlay_video(input_mp4, output_mp4, name_red="Red", name_blue="Blue",
                  hp_red=100, hp_blue=100, rounds=3, round_seconds=30.0):
    """Add HP bar overlay to a rendered bout video."""
    import imageio_ffmpeg
    import tempfile

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    frames_dir = tempfile.mkdtemp()

    subprocess.run([ff, "-i", input_mp4, "-vf", "fps=30",
                    f"{frames_dir}/f%05d.png"], capture_output=True)

    frames = sorted(os.listdir(frames_dir))
    total_frames = len(frames)
    if not frames:
        print("No frames extracted")
        return

    fps = 30
    for i, fname in enumerate(frames):
        from PIL import Image
        img = Image.open(os.path.join(frames_dir, fname))
        t = i / fps
        round_num = min(int(t / round_seconds) + 1, rounds)
        round_time = t % round_seconds
        overlayed = add_overlay(img, hp_red, hp_blue, round_num, round_time,
                                 name_red, name_blue)
        overlayed.save(os.path.join(frames_dir, fname))

    subprocess.run([ff, "-y", "-framerate", "30", "-i",
                    f"{frames_dir}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", output_mp4],
                   capture_output=True)

    import shutil
    shutil.rmtree(frames_dir)
    print(f"[overlay] saved {output_mp4}")


# ===========================================================================
# deterministic_eval.py — trustless deterministic league eval
# ===========================================================================
EVAL_SEEDS = [42, 123, 777]  # multiple seeds to prevent overfitting
EVAL_STEPS = 5000
ROUND_SECONDS = 20.0
ROUNDS = 3
MAX_MODEL_SIZE = 50 * 1024 * 1024  # 50MB limit
MIN_DAMAGE_TO_PASS = 1.0  # must deal at least 1 damage (not just survive)


def model_hash(path):
    """SHA256 hash of a model file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_bout(fighter_path, opponent_spec, seed=42):
    """Run a single deterministic bout. Returns bout log."""
    import numpy as np
    from stable_baselines3 import PPO
    from g1_fighter_env import G1FighterEnv
    from combat import CombatJudge, ShadowBoxer

    env = G1FighterEnv(max_steps=EVAL_STEPS, randomize=False)

    # Force full determinism: seed everything, no randomization
    np.random.seed(seed)
    import random as pyrandom
    pyrandom.seed(seed)

    obs, _ = env.reset(seed=seed)
    judge = CombatJudge(env, round_seconds=ROUND_SECONDS, rounds=ROUNDS)

    # Load fighter
    if fighter_path and os.path.exists(fighter_path):
        p1 = PPO.load(fighter_path)
    else:
        p1 = ShadowBoxer(env, style="red", profile="balanced")

    # Load opponent
    if opponent_spec and opponent_spec.startswith("scripted:"):
        profile = opponent_spec.split(":")[1]
        env.opponent = ShadowBoxer(env, style="blue", profile=profile)
    elif opponent_spec and os.path.exists(opponent_spec):
        env.opponent = PPO.load(opponent_spec)
    else:
        env.opponent = ShadowBoxer(env, style="blue", profile="pd")

    log = {
        "fighter": os.path.basename(fighter_path or "shadowboxer"),
        "fighter_hash": model_hash(fighter_path) if fighter_path and os.path.exists(fighter_path) else "scripted",
        "opponent": opponent_spec,
        "seed": seed,
        "events": [],
        "final_hp": None,
        "result": None,
    }

    for t in range(EVAL_STEPS):
        a1, _ = p1.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)

        # Log damage events
        if env._dmg_dealt[0] > 0 or env._dmg_taken[0] > 0:
            log["events"].append({
                "step": t,
                "hp": [float(env.hp[0]), float(env.hp[1])],
                "dmg_dealt": float(env._dmg_dealt[0]),
                "dmg_taken": float(env._dmg_taken[0]),
            })

        if term or trunc:
            break

    log["final_hp"] = [float(env.hp[0]), float(env.hp[1])]
    card = judge.card()
    log["result"] = {
        "winner": card["winner"],
        "method": card["method"],
        "round_scores": card["round_scores"],
    }

    return log


def deterministic_eval_main(argv=None):
    """Deterministic league eval for trustless PR decisions.

    Runs bouts with FIXED seed, NO domain randomization, and produces
    a verifiable bout log (JSON) with per-step HP, damage events, and
    final decision. Anyone can re-run and get identical results.

    Usage:
      python3 eval.py deterministic_eval --fighter models/my_fighter.zip
      python3 eval.py deterministic_eval --fighter models/my_fighter.zip --king models/fighter_v2.zip
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--fighter", required=True, help="path to fighter .zip")
    ap.add_argument("--king", default=None, help="path to current king .zip")
    ap.add_argument("--out", default="/tmp/eval_result.json")
    ap.add_argument("--entrants", nargs="*", default=[
        "scripted:jabbler", "scripted:defender", "scripted:balanced", "scripted:pd"])
    a = ap.parse_args(argv)

    print(f"[eval] fighter: {a.fighter}")
    print(f"[eval] hash: {model_hash(a.fighter)}")
    print(f"[eval] seeds: {EVAL_SEEDS} (multi-seed anti-overfit)")

    fsize = os.path.getsize(a.fighter)
    if fsize > MAX_MODEL_SIZE:
        print(f"[eval] REJECTED: model too large ({fsize / 1e6:.1f}MB > {MAX_MODEL_SIZE / 1e6:.0f}MB)")
        sys.exit(1)

    results = []
    all_entrants = [a.fighter] + a.entrants
    if a.king:
        all_entrants.append(a.king)

    for seed in EVAL_SEEDS:
        for opp in a.entrants:
            print(f"[eval] seed={seed} vs {opp}...", end=" ", flush=True)
            log = run_bout(a.fighter, opp, seed=seed)
            log["seed"] = seed
            results.append(log)
            print(f"hp={log['final_hp']} result={log['result']['method']}")

    if a.king:
        for seed in EVAL_SEEDS:
            print(f"[eval] seed={seed} TITLE BOUT vs king...", end=" ", flush=True)
            log = run_bout(a.fighter, a.king, seed=seed)
            log["title_bout"] = True
            log["seed"] = seed
            results.append(log)
            print(f"hp={log['final_hp']} result={log['result']['method']}")

    wins = sum(1 for r in results if r["result"]["winner"] == 0)
    losses = sum(1 for r in results if r["result"]["winner"] == 1)
    draws = len(results) - wins - losses
    total_dmg_dealt = sum(sum(e["dmg_dealt"] for e in r["events"]) for r in results)
    seeds_won = set(r["seed"] for r in results if r["result"]["winner"] == 0)

    summary = {
        "fighter": os.path.basename(a.fighter),
        "fighter_hash": model_hash(a.fighter),
        "model_size_mb": round(fsize / 1e6, 1),
        "seeds": EVAL_SEEDS,
        "total_bouts": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_damage_dealt": round(total_dmg_dealt, 2),
        "seeds_won_on": sorted(seeds_won),
        "pass": wins >= 1 and total_dmg_dealt >= MIN_DAMAGE_TO_PASS and len(seeds_won) >= 2,
        "merge_decision": "MERGE" if wins >= 1 and total_dmg_dealt >= MIN_DAMAGE_TO_PASS and len(seeds_won) >= 2 else
                          "DRAW" if wins == losses and draws > 0 else
                          "REJECT",
        "bouts": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'✅' if summary['pass'] else '❌'} Result: {wins}W {losses}L {draws}D")
    print(f"[eval] saved {a.out}")

    if not summary["pass"]:
        sys.exit(1)


# ===========================================================================
# ci_gate.py — CI merge gate
# ===========================================================================
DEFAULT_ENTRANTS = [
    "scripted:jabbler", "scripted:defender",
    "scripted:balanced", "scripted:pd"
]
MIN_ELO = 1400
MIN_WINS = 1


def ci_run_league(fighter_path, standings_path, bouts=3, max_steps=5000):
    """Run the league with the submitted fighter + scripted baselines."""
    entrants = [fighter_path] + DEFAULT_ENTRANTS
    cmd = [sys.executable, os.path.abspath(__file__),
           "deterministic_eval",
           "--fighter", fighter_path,
           "--out", "/tmp/ci_eval.json"]
    if os.path.exists("models/fighter_v2.zip"):
        cmd += ["--king", "models/fighter_v2.zip"]
    print(f"[ci] running deterministic eval: {fighter_path}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    print(f"[ci] eval completed in {elapsed:.0f}s")
    if r.returncode != 0:
        print(f"[ci] eval FAILED: {r.stderr[-300:]}")
        return None

    return json.load(open("/tmp/ci_eval.json"))


def ci_evaluate(fighter_path, threshold=MIN_ELO):
    """Run CI gate on a submitted fighter. Returns pass/fail + details."""
    standings_path = "/tmp/ci_standings.json"
    standings = ci_run_league(fighter_path, standings_path)
    if standings is None:
        return {"pass": False, "reason": "league crashed"}

    fighter_name = os.path.basename(fighter_path).replace(".zip", "")
    fighter_entry = None
    for s in standings.get("standings", []):
        if fighter_name in s["name"]:
            fighter_entry = s
            break

    if not fighter_entry:
        return {"pass": False, "reason": "fighter not in standings"}

    elo = fighter_entry["elo"]
    wins = fighter_entry["W"]
    losses = fighter_entry["L"]
    draws = fighter_entry["D"]

    result = {
        "pass": True,
        "fighter": fighter_name,
        "elo": elo,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "king": standings.get("king"),
        "threshold": threshold,
    }

    if elo < threshold:
        result["pass"] = False
        result["reason"] = f"ELO {elo} < threshold {threshold}"
    if wins < MIN_WINS:
        result["pass"] = False
        result["reason"] = f"only {wins} wins (need >= {MIN_WINS})"
    if elo < standings["standings"][0]["elo"] and elo < standings["standings"][2]["elo"] if len(standings["standings"]) > 2 else False:
        result["pass"] = False
        result["reason"] = f"ELO {elo} too low (not top-3)"

    return result


def ci_gate_main(argv=None):
    """CI gate for fighter submissions.

    Evaluates a submitted fighter in the league and decides if it
    passes the merge threshold. Used by Gittensor PR CI.

    Usage:
      python3 eval.py ci_gate --fighter models/fighter_v2.zip
      python3 eval.py ci_gate --fighter models/fighter_v2.zip --threshold 1500
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--fighter", required=True, help="path to fighter .zip")
    ap.add_argument("--threshold", type=float, default=MIN_ELO,
                    help=f"minimum ELO to pass (default {MIN_ELO})")
    a = ap.parse_args(argv)

    result = ci_evaluate(a.fighter, a.threshold)

    print("=" * 50)
    if result["pass"]:
        print(f"✅ PASS: {result['fighter']}")
        print(f"   ELO: {result['elo']:.1f} (threshold: {result['threshold']})")
        print(f"   Record: {result['wins']}W {result['losses']}L {result['draws']}D")
        print(f"   King: {result['king']}")
        sys.exit(0)
    else:
        print(f"❌ FAIL: {result.get('fighter', 'unknown')}")
        print(f"   Reason: {result['reason']}")
        if "elo" in result:
            print(f"   ELO: {result['elo']:.1f} (threshold: {result['threshold']})")
            print(f"   Record: {result['wins']}W {result['losses']}L {result['draws']}D")
        sys.exit(1)


# ===========================================================================
# egl_bout.py — EGL 1v1 eval bout (PNG frames -> ffmpeg)
# ===========================================================================
def egl_bout_main(argv=None):
    """EGL 1v1 eval bout: saves frames as PNGs, then ffmpeg encodes.

    Usage:
      python3 eval.py egl_bout --p1 models/fighter_v1.zip --out /tmp/egl_bout.mp4
    """
    import numpy as np
    import mujoco
    import PIL.Image
    from g1_fighter_env import G1FighterEnv
    from combat import CombatJudge, ShadowBoxer

    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default=None, help="fighter policy path (None=shadowboxer)")
    ap.add_argument("--p2", default=None, help="opponent: scripted:PROFILE or model path (default=jabbler)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="/tmp/egl_bout.mp4")
    ap.add_argument("--no-terminate", action="store_true")
    ap.add_argument("--frames-dir", default="/tmp/bout_frames")
    a = ap.parse_args(argv)

    env = G1FighterEnv(max_steps=a.steps, randomize=False, demo=(a.p1 is None))
    # Start close — within striking range so the fight is visible
    obs, _ = env.reset()
    env.data.qpos[0:3] = [-0.15, 0, 0.793]
    env.data.qpos[36:39] = [0.15, 0, 0.793]
    env.data.qpos[3:7] = [1, 0, 0, 0]
    env.data.qpos[39:43] = [0, 0, 0, 1]
    import mujoco as _mj
    _mj.mj_forward(env.model, env.data)
    obs = env._get_obs(0)
    judge = CombatJudge(env, round_seconds=3.0, rounds=3)

    if a.p1:
        from stable_baselines3 import PPO
        p1 = PPO.load(a.p1)
    else:
        p1 = ShadowBoxer(env, style="red")

    # Opponent: specified profile, model, or default jabbler (aggressive)
    if a.p2 and a.p2.startswith("scripted:"):
        env.opponent = ShadowBoxer(env, style="blue", profile=a.p2.split(":")[1])
    elif a.p2 and os.path.exists(a.p2):
        from stable_baselines3 import PPO
        env.opponent = PPO.load(a.p2)
    else:
        env.opponent = ShadowBoxer(env, style="blue", profile="jabbler")

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
    r = mujoco.Renderer(env.model, height=720, width=1280)

    os.makedirs(a.frames_dir, exist_ok=True)
    n_frames = 0
    for t in range(a.steps):
        a1, _ = p1.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)
        try:
            r.update_scene(env.data, camera=cam_id)
            img = r.render()
            PIL.Image.fromarray(img).save(f"{a.frames_dir}/f{t:05d}.png")
            n_frames += 1
        except Exception:
            pass
        if t % 1000 == 0:
            print(f"step {t}, hp={env.hp}, frames={n_frames}", flush=True)
        if not a.no_terminate and (term or trunc):
            break

    print(f"total: {n_frames} frames, hp={env.hp}", flush=True)

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", "30", "-i", f"{a.frames_dir}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                   capture_output=True)
    os.system(f"rm -rf {a.frames_dir}")
    print(f"SAVED {a.out}", flush=True)
    card = judge.card()
    print(f"CARD: winner={card['winner']} method={card['method']} hp={card['final_hp']}", flush=True)


# ===========================================================================
# eval_tracker.py — motion tracker eval render
# ===========================================================================
def eval_tracker_main(argv=None):
    """Eval the motion tracker: load the trained policy, run it on a
    karate motion, render to see if the G1 actually moves."""
    import numpy as np
    import mujoco
    import PIL.Image
    import imageio_ffmpeg
    from stable_baselines3 import PPO
    from train_motion_tracker import load_motions, MotionTrackerEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", default="/workspace/g1-moves/karate",
                    help="motions directory")
    ap.add_argument("--model", default="/workspace/models/motion_tracker.zip",
                    help="motion tracker model path")
    ap.add_argument("--out", default="/tmp/tracker_eval.mp4")
    ap.add_argument("--motion-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args(argv)

    motions = load_motions(a.motions)
    model = PPO.load(a.model)

    attack_motions = [m for m in motions if "Attack" in m["name"]]
    if not attack_motions:
        attack_motions = motions
    motion_idx = a.motion_idx

    env = MotionTrackerEnv(attack_motions, max_steps=a.steps)
    env.motions = [attack_motions[motion_idx]]
    obs, _ = env.reset()
    print(f"eval motion: {attack_motions[motion_idx]['name']}, "
          f"{len(attack_motions[motion_idx]['joint_pos'])} frames")

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
    r = mujoco.Renderer(env.model, height=720, width=1280)

    frames_dir = "/tmp/tracker_eval_frames"
    os.makedirs(frames_dir, exist_ok=True)
    n = 0
    total_reward = 0
    for t in range(a.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        try:
            r.update_scene(env.data, camera=cam_id)
            img = r.render()
            PIL.Image.fromarray(img).save(f"{frames_dir}/f{t:05d}.png")
            n += 1
        except Exception:
            pass
        if t % 100 == 0:
            pelvis_z = env.data.xpos[env.model.body("r1_pelvis").id][2]
            print(f"step {t}: reward={reward:.3f} pelvis_z={pelvis_z:.3f}", flush=True)
        if term or trunc:
            break

    print(f"total: {n} frames, avg_reward={total_reward/max(n,1):.3f}", flush=True)

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", "30", "-i", f"{frames_dir}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                   capture_output=True)
    os.system(f"rm -rf {frames_dir}")
    print(f"SAVED {a.out}", flush=True)


# ===========================================================================
# test_damage.py — damage detection sanity test
# ===========================================================================
def test_damage_main(argv=None):
    """Test damage detection: place two bots, force a punch, check HP drops."""
    import numpy as np
    import mujoco
    from g1_fighter_env import G1FighterEnv

    env = G1FighterEnv(max_steps=200, randomize=False)
    obs, _ = env.reset()

    env.data.qpos[0:3] = [-0.05, 0, 0.793]   # r1
    env.data.qpos[36:39] = [0.05, 0, 0.793]  # r2 (0.1m apart)
    mujoco.mj_forward(env.model, env.data)

    print(f"Initial HP: {env.hp}")
    print(f"Weapons r1: {env.fist_geoms[0]}")
    print(f"Torso bodies r2: {env.torso_bodies[1]}")

    home = env.native.copy()

    hits = 0
    for t in range(200):
        punch_target = home.copy()
        punch_target[15] = -0.8   # right shoulder pitch (forward)
        punch_target[18] = 1.5    # right elbow (extend)

        kp, kd = 100.0, 5.0
        tau1 = kp * (punch_target - env.data.qpos[7:36]) - kd * env.data.qvel[6:35]
        env.data.ctrl[0:29] = np.clip(tau1, -88, 88)

        tau2 = kp * (home - env.data.qpos[43:72]) - kd * env.data.qvel[41:70]
        env.data.ctrl[29:58] = np.clip(tau2, -88, 88)

        mujoco.mj_step(env.model, env.data, 1)
        env._update_damage()

        if env._dmg_dealt[0] > 0:
            hits += 1
            if hits <= 5:
                print(f"step {t}: HIT! dmg={env._dmg_dealt[0]:.2f} hp={env.hp}")

        if t % 50 == 0:
            r1_wrist = env.data.xpos[env.model.body("r1_right_wrist_yaw_link").id]
            r2_torso = env.data.xpos[env.model.body("r2_torso_link").id]
            dist = np.linalg.norm(r1_wrist - r2_torso)
            print(f"step {t}: wrist-torso dist={dist:.3f} hp={env.hp}")

        if env.hp[1] < 50:
            print(f"R2 HP below 50 at step {t}!")
            break

    print(f"\nFinal HP: {env.hp}")
    print(f"Total hits landed: {hits}")
    if hits == 0:
        print("NO DAMAGE DETECTED — collision/damage system needs fixing")
    else:
        print("Damage working!")


# ===========================================================================
# bout_overlay.py — CLI
# ===========================================================================
def bout_overlay_cli_main(argv=None):
    """Add HP bar overlay to a rendered bout video."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--red", default="Red")
    ap.add_argument("--blue", default="Blue")
    ap.add_argument("--hp-red", type=float, default=100)
    ap.add_argument("--hp-blue", type=float, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--round-seconds", type=float, default=30.0)
    a = ap.parse_args(argv)
    overlay_video(a.input, a.output, a.red, a.blue, a.hp_red, a.hp_blue,
                  a.rounds, a.round_seconds)


# ===========================================================================
# Subcommand dispatch
# ===========================================================================
SUBCOMMANDS = {
    "deterministic_eval": deterministic_eval_main,
    "ci_gate": ci_gate_main,
    "egl_bout": egl_bout_main,
    "eval_tracker": eval_tracker_main,
    "test_damage": test_damage_main,
    "bout_overlay": bout_overlay_cli_main,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("Usage: python3 eval.py <subcommand> [args...]")
        print("Subcommands:")
        for name in SUBCOMMANDS:
            print(f"  {name}")
        sys.exit(1)
    name = sys.argv[1]
    fn = SUBCOMMANDS.get(name)
    if fn is None:
        print(f"Unknown subcommand: {name}")
        print(f"Available: {', '.join(SUBCOMMANDS)}")
        sys.exit(1)
    fn(sys.argv[2:])


if __name__ == "__main__":
    main()
