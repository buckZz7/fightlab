"""King-of-the-hill league: round-robin bouts + ELO standings,
page generation, auto-update, and bout rendering.

This file consolidates (preserving all functionality):
  - league.py             : round-robin bouts + ELO scoring
  - gen_league_page.py    : league standings HTML page (docs/index.html)
  - league_update.py      : auto-update (run league + render + page) cron entry
  - render_league_bouts.py: render league bouts to MP4s (EGL + tracking cam)

Entrant types (in --entrants):
  models/fighter_v1        -> trained PPO policy (red loads it)
  scripted:jabbler         -> scripted aggressive ShadowBoxer
  scripted:defender        -> scripted guard-heavy ShadowBoxer
  scripted:pd              -> PD-to-HOME baseline (random arm, stands)

Usage:
  python3 league.py --entrants models/fighter_v1 scripted:jabbler \
      scripted:defender scripted:pd --pd --bouts 3 \
      --out league_standings.json
  python3 league.py page --standings docs/league_test.json --out docs/index.html
  python3 league.py update --standings docs/league_standings.json
  python3 league.py render --standings docs/league_standings.json --pd --steps 5000
"""
import os, sys, argparse, json, itertools, datetime, html, glob, subprocess
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from stable_baselines3 import PPO

from g1_fighter_env import G1FighterEnv
from combat import CombatJudge, ShadowBoxer


# ===========================================================================
# Round-robin + ELO (was league.py main)
# ===========================================================================
def _elo(ra, rb, score_a):
    """score_a = 1.0 win, 0.5 draw, 0.0 loss. K=32."""
    k = 32.0
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


class _RandomPD:
    """PD baseline: random arm residual (still stands via PD legs)."""
    def __init__(self, env):
        self.env = env
    def predict(self, obs, deterministic=True):
        return self.env.action_space.sample(), None


def _load_entrant(spec, env, for_blue=False):
    """Return a callable predict(obs)->(action,None) for the entrant.
    for_blue=True -> loaded as the env's opponent (r2)."""
    if spec.startswith("scripted:"):
        style = spec.split(":", 1)[1]
        if style == "pd":
            return _RandomPD(env)
        sb_style = "blue" if for_blue else "red"
        profile = style if style in ("jabbler", "defender") else "balanced"
        return ShadowBoxer(env, style=sb_style, profile=profile)
    return PPO.load(spec)


def run_bout(name_a, spec_a, name_b, spec_b, balance, max_steps,
              round_seconds, rounds):
    """One bout: A (red) vs B (blue). balance=None -> PD substrate.
    Blue is loaded as the env's opponent (r2). Returns
    (winner_name or None, method, hp, n_rounds)."""
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=max_steps, randomize=False)
    env.opponent = _load_entrant(spec_b, env, for_blue=True)
    judge = CombatJudge(env, round_seconds=round_seconds, rounds=rounds)
    red = _load_entrant(spec_a, env, for_blue=False)
    obs, _ = env.reset()
    done = False
    t = 0
    while not done and t < max_steps:
        a1 = red.predict(obs, deterministic=True)[0]
        obs, rew, term, trunc, info = judge.step(a1)
        done = term or trunc or judge.ko or (judge.winner is not None)
        t += 1
    card = judge.card()
    w = card["winner"]
    winner = name_a if w == 0 else (name_b if w == 1 else None)
    return winner, card["method"], card["final_hp"], len(card["round_scores"])


def league_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrants", nargs="+", required=True,
                    help="fighter model paths (e.g. models/fighter_a)")
    ap.add_argument("--balance", default=None,
                    help="balance model path; omit for PD substrate")
    ap.add_argument("--pd", action="store_true",
                    help="use PD-to-HOME substrate")
    ap.add_argument("--bouts", type=int, default=3,
                    help="bouts per pair (round-robin)")
    ap.add_argument("--max_steps", type=int, default=900)
    ap.add_argument("--round_seconds", type=float, default=30.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="league_standings.json")
    a = ap.parse_args(argv)

    balance = None if a.pd else a.balance
    entrants = {os.path.basename(p): p for p in a.entrants}
    names = list(entrants)
    elo = {n: 1500.0 for n in names}
    record = {n: {"W": 0, "L": 0, "D": 0} for n in names}

    print(f"[league] {len(names)} entrants, {a.bouts} bouts/pair, "
          f"substrate={'PD' if a.pd else balance}")
    results = []
    for na, nb in itertools.combinations(names, 2):
        for b in range(a.bouts):
            w, method, hp, nr = run_bout(na, entrants[na], nb, entrants[nb],
                                           balance, a.max_steps,
                                           a.round_seconds, a.rounds)
            if w == na:
                elo[na], elo[nb] = _elo(elo[na], elo[nb], 1.0)
                record[na]["W"] += 1; record[nb]["L"] += 1
                score = "A win"
            elif w == nb:
                elo[na], elo[nb] = _elo(elo[na], elo[nb], 0.0)
                record[na]["L"] += 1; record[nb]["W"] += 1
                score = "B win"
            else:
                elo[na], elo[nb] = _elo(elo[na], elo[nb], 0.5)
                record[na]["D"] += 1; record[nb]["D"] += 1
                score = "draw"
            results.append({"red": na, "blue": nb, "bout": b + 1,
                             "winner": w, "method": method,
                             "hp": hp, "rounds": nr,
                             "elo_red": round(elo[na], 1),
                             "elo_blue": round(elo[nb], 1)})
            print(f"  {na} vs {nb} #{b+1}: {score} ({method}) "
                  f"hp={hp} elo={elo[na]:.0f}/{elo[nb]:.0f}")

    standings = sorted(
        [{"name": n, "elo": round(elo[n], 1), **record[n]}
         for n in names],
        key=lambda x: -x["elo"])
    king = standings[0]["name"] if standings else None
    out = {
        "king": king,
        "standings": standings,
        "results": results,
        "substrate": "PD" if a.pd else balance,
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[league] KING: {king}  elo={standings[0]['elo'] if standings else 0}")
    print(f"[league] standings -> {a.out}")
    for s in standings:
        print(f"   {s['elo']:6.1f}  {s['name']:24s}  "
              f"W{s['W']} L{s['L']} D{s['D']}")


# ===========================================================================
# gen_league_page.py — league standings HTML page
# ===========================================================================
def page(d, bout_map, title_video="bouts/title_bout.mp4"):
    king = d.get("king") or "TBD"
    standings = d.get("standings", [])
    results = d.get("results", [])
    sub = d.get("substrate", "PD")
    gen = d.get("generated_utc", "")

    rows = ""
    for i, s in enumerate(standings, 1):
        medal = {1: "1", 2: "2", 3: "3"}.get(i, f"{i}")
        crown = "" if s["name"] == king else ""
        rows += f"""
        <tr class="{'king-row' if s['name']==king else ''}">
          <td class="pos">{medal}</td>
          <td class="name">{html.escape(s['name'])}{crown}</td>
          <td class="elo">{s['elo']:.0f}</td>
          <td>{s['W']}</td><td>{s['L']}</td><td>{s['D']}</td>
        </tr>"""

    vids = ""
    for r in results:
        mp4 = bout_map.get((r["red"], r["blue"]))
        if not mp4:
            continue
        w = r.get("winner") or "Draw"
        vids += f"""
      <div class="bout">
        <div class="bout-head">
          <span class="red">{html.escape(r['red'])}</span>
          <span class="vs">vs</span>
          <span class="blue">{html.escape(r['blue'])}</span>
        </div>
        <video controls muted playsinline loop>
          <source src="{mp4}" type="video/mp4">
        </video>
        <div class="bout-meta">
          <b>{html.escape(str(w))}</b> &middot; {html.escape(str(r.get('method','')))}
          &middot; HP {r.get('hp', '')} &middot; {r.get('rounds','?')} rounds
        </div>
      </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FightLab &mdash; King of the Hill</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;700&family=Anton&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0a0a; --bg2: #111111; --card: #161616; --line: #222222;
    --mut: #666666; --red: #e74c3c; --red-dim: #c0392b;
    --gold: #d4af37; --txt: #f0f0f0;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{
    background:var(--bg); color:var(--txt);
    font-family:'Oswald',sans-serif; font-weight:400;
    max-width:900px; margin:0 auto; padding:0 24px 60px;
  }}

  /* Nav back to landing */
  .nav {{
    padding:20px 0; border-bottom:1px solid var(--line);
    display:flex; justify-content:space-between; align-items:center;
  }}
  .nav a {{
    font-family:'Anton',sans-serif; font-size:22px;
    text-transform:uppercase; text-decoration:none;
    color:var(--txt); letter-spacing:0.02em;
  }}
  .nav a .red {{ color:var(--red); }}
  .nav .back {{
    font-family:'Oswald',sans-serif; font-size:13px;
    color:var(--mut); text-transform:uppercase; letter-spacing:0.1em;
  }}
  .nav .back:hover {{ color:var(--red); }}

  h1 {{
    font-family:'Anton',sans-serif; font-size:48px;
    text-transform:uppercase; letter-spacing:0.02em;
    margin:32px 0 4px; line-height:0.9;
  }}
  h1 .red {{ color:var(--red); }}
  .sub {{
    color:var(--mut); font-size:14px;
    text-transform:uppercase; letter-spacing:0.2em;
    margin-bottom:24px;
  }}

  /* King banner */
  .king-banner {{
    margin:0 0 24px; padding:28px; border:1px solid var(--red-dim);
    border-left:4px solid var(--red);
    background:linear-gradient(135deg, rgba(231,76,60,0.08), transparent);
  }}
  .king-banner .lbl {{
    color:var(--red); font-size:12px;
    text-transform:uppercase; letter-spacing:0.3em;
    font-family:'JetBrains Mono',monospace;
  }}
  .king-banner .who {{
    font-family:'Anton',sans-serif; font-size:32px;
    text-transform:uppercase; margin-top:6px; color:var(--gold);
  }}

  /* Standings table */
  .card {{
    background:var(--card); border:1px solid var(--line);
    padding:24px; margin:16px 0;
  }}
  h2 {{
    font-family:'Anton',sans-serif; font-size:28px;
    text-transform:uppercase; margin-bottom:16px;
    letter-spacing:0.02em;
  }}
  h2 .red {{ color:var(--red); }}

  table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums }}
  th, td {{
    text-align:left; padding:12px 10px;
    border-bottom:1px solid var(--line);
    font-family:'Oswald',sans-serif;
  }}
  th {{
    color:var(--mut); font-size:11px; text-transform:uppercase;
    letter-spacing:0.2em; font-weight:500;
  }}
  td.elo {{ font-weight:700; color:var(--gold); font-family:'JetBrains Mono',monospace; }}
  td.pos {{ width:40px; font-family:'Anton',sans-serif; font-size:20px; color:var(--mut); }}
  .king-row td {{ background:rgba(231,76,60,0.05); }}
  .king-row td.elo {{ color:var(--gold); }}

  /* Bouts */
  .bouts {{ display:grid; grid-template-columns:1fr; gap:16px }}
  .bout {{
    background:var(--card); border:1px solid var(--line);
    overflow:hidden;
  }}
  .bout-head {{
    padding:16px; display:flex; gap:12px;
    align-items:center; border-bottom:1px solid var(--line);
    font-family:'Oswald',sans-serif; font-weight:700;
    text-transform:uppercase; font-size:15px;
  }}
  .bout-head .red {{ color:var(--red) }}
  .bout-head .blue {{ color:#3498db }}
  .bout-head .vs {{ color:var(--mut); font-weight:400; font-size:12px; letter-spacing:0.2em; }}
  .bout video {{ width:100%; display:block; background:#000; aspect-ratio:16/9 }}
  .bout-meta {{ padding:12px 16px; color:var(--mut); font-size:13px; }}

  footer {{
    margin-top:40px; padding:20px 0; border-top:1px solid var(--line);
    text-align:center; color:var(--mut); font-size:12px;
    text-transform:uppercase; letter-spacing:0.1em;
  }}
  footer a {{ color:var(--red); text-decoration:none; }}
  a {{ color:var(--red); text-decoration:none; }}
</style>
</head>
<body>
  <div class="nav">
    <a href="index.html">Fight<span class="red">Lab</span></a>
    <a href="index.html" class="back">Back</a>
  </div>
  <h1>King of the <span class="red">Hill</span></h1>
  <div class="sub">Autonomous humanoid combat league &middot; MuJoCo + RL
   &middot; substrate: {html.escape(str(sub))}</div>

  <div class="king-banner">
    <div class="lbl">Current King</div>
    <div class="who">{html.escape(str(king))}</div>
  </div>

  <div class="card">
    <h2>Standings</h2>
    <table>
      <thead><tr><th>#</th><th>Fighter</th><th>ELO</th>
      <th>W</th><th>L</th><th>D</th></tr></thead>
      <tbody>{rows}
      </tbody>
    </table>
  </div>

  <h2 style="margin-top:26px">Bouts</h2>
  <div class="bouts">{vids if vids else '<div class="sub">No bout videos rendered yet.</div>'}
  </div>

  <h2 style="margin-top:26px">Title Bout</h2>
  <div class="bout">
    <div class="bout-head">
      <span class="red">King: {html.escape(str(king))}</span>
      <span class="vs">vs</span>
      <span class="blue">Challenger</span>
    </div>
    <video controls muted playsinline loop>
      <source src="{title_video}" type="video/mp4">
    </video>
  </div>

  <footer>
    Generated {html.escape(gen)} UTC &middot; FightLab &middot;
    <a href="https://github.com/buckZz7/fightlab">github.com/buckZz7/fightlab</a>
  </footer>
</body>
</html>"""


def gen_page_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", default="docs/league_test.json")
    ap.add_argument("--out", default="docs/league.html")
    a = ap.parse_args(argv)
    d = json.load(open(a.standings))
    bout_map = {}
    for r in d.get("results", []):
        if r.get("mp4"):
            bout_map[(r["red"], r["blue"])] = r["mp4"]
    bouts_dir = os.path.join(os.path.dirname(a.out), "bouts")
    title_video = "bouts/title_bout.mp4"
    if os.path.exists(bouts_dir):
        title_files = sorted([f for f in os.listdir(bouts_dir)
                              if f.startswith("title_cycle") and f.endswith(".mp4")])
        if title_files:
            title_video = f"bouts/{title_files[-1]}"
        elif os.path.exists(os.path.join(bouts_dir, "title_bout.mp4")):
            title_video = "bouts/title_bout.mp4"

    html_txt = page(d, bout_map, title_video)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html_txt)
    print(f"[page] wrote {a.out} ({len(html_txt)} chars), "
          f"{len(bout_map)} bouts with video")


# ===========================================================================
# render_league_bouts.py — render league bouts to MP4 (EGL + tracking cam)
# ===========================================================================
def render_bout(spec_a, spec_b, balance, out, steps):
    """Render a single bout to MP4 using EGL + tracking camera."""
    import mujoco
    import PIL.Image
    import imageio_ffmpeg
    env = G1FighterEnv(balance_path=balance, opponent_path=None,
                       max_steps=steps, randomize=False)
    env.opponent = _load_entrant(spec_b, env, for_blue=True)
    judge = CombatJudge(env, round_seconds=30.0, rounds=3)
    red = _load_entrant(spec_a, env, for_blue=False)

    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "broadcast")
    r = mujoco.Renderer(env.model, height=720, width=1280)

    frames_dir = out.replace(".mp4", "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    obs, _ = env.reset()
    n = 0
    for t in range(steps):
        a1, _ = red.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = judge.step(a1)
        try:
            r.update_scene(env.data, camera=cam_id)
            img = r.render()
            PIL.Image.fromarray(img).save(f"{frames_dir}/f{t:05d}.png")
            n += 1
        except Exception:
            pass
        if term or trunc:
            break

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-framerate", "30", "-i",
                    f"{frames_dir}/f%05d.png", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", out],
                   capture_output=True)
    os.system(f"rm -rf {frames_dir}")
    print(f"[render] {out} ({n} frames) hp={env.hp}")


def render_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out-dir", default="docs/bouts")
    ap.add_argument("--max-bouts", type=int, default=4)
    ap.add_argument("--pd", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    standings = json.load(open(a.standings))
    bouts = standings.get("bouts", [])[:a.max_bouts]

    for bout in bouts:
        name_a = bout["a"]
        name_b = bout["b"]
        spec_a = bout.get("spec_a", name_a)
        spec_b = bout.get("spec_b", name_b)
        fname = f"{name_a}_vs_{name_b}".replace(" ", "_").replace(":", "_")
        out = os.path.join(a.out_dir, f"{fname}.mp4")
        print(f"[render] {name_a} vs {name_b} -> {out}")
        try:
            render_bout(spec_a, spec_b, None, out, a.steps)
        except Exception as e:
            print(f"  [error] {e}")


# ===========================================================================
# league_update.py — auto-update pipeline (cron entry)
# ===========================================================================
def _sh(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  [warn]", (r.stderr or r.stdout)[-400:])
    return r


def update_main(argv=None):
    """Auto-update the king-of-the-hill league page when a trained
    fighter lands. One-shot: runs the league (mixing trained fighters
    + scripted reference), renders bouts, regenerates the page, and
    reports a summary. Intended to be driven by a cron that watches
    for models/fighter_v1.zip (or newer checkpoints) on the pod.

    Usage:
      python3 league.py update --standings docs/league_standings.json
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", default="docs/league_standings.json")
    ap.add_argument("--bouts", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--render-steps", type=int, default=300)
    ap.add_argument("--max-render-bouts", type=int, default=4)
    a = ap.parse_args(argv)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    trained = sorted(glob.glob("models/fighter_*.zip"))
    trained = [p[:-4] for p in trained]  # strip .zip
    entrants = trained + [
        "scripted:jabbler", "scripted:defender", "scripted:balanced", "scripted:pd"]
    print(f"[league] entrants: {entrants}")

    # 1) run the league
    _sh([sys.executable, __file__,
        "--entrants", *entrants,
        "--pd", "--bouts", str(a.bouts),
        "--max_steps", str(a.max_steps), "--round_seconds", "20",
        "--out", a.standings])

    # 2) render top bouts
    _sh([sys.executable, __file__, "render",
        "--standings", a.standings, "--pd",
        "--steps", str(a.render_steps), "--out-dir", "docs/bouts",
        "--max-bouts", str(a.max_render_bouts)])

    # 3) regenerate the page
    _sh([sys.executable, __file__, "page",
        "--standings", a.standings, "--out", "docs/index.html"])

    # 4) report
    d = json.load(open(a.standings))
    king = d.get("king")
    print(f"[done] king={king}")
    for s in d.get("standings", [])[:3]:
        print(f"  {s['elo']:6.1f} {s['name']:24s} W{s['W']} L{s['L']} D{s['D']}")


# ===========================================================================
# Subcommand dispatch
# ===========================================================================
SUBCOMMANDS = {
    "page": gen_page_main,
    "render": render_main,
    "update": update_main,
}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in SUBCOMMANDS:
        # Subcommand dispatch (page / render / update)
        SUBCOMMANDS[sys.argv[1]](sys.argv[2:])
        return
    # Default: run the round-robin league
    league_main(sys.argv[1:])


if __name__ == "__main__":
    main()
