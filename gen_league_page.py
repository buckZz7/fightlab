"""Generate the king-of-the-hill league page (docs/index.html).

Reads league_standings.json (+ rendered bout MP4 paths) and emits a
self-contained dark-themed page: king banner, standings table (ELO,
W/L/D), embedded bout videos.

Usage:
  python3 gen_league_page.py --standings docs/league_test.json \
      --out docs/index.html
"""
import os, sys, argparse, json, datetime, html


def page(d, bout_map, title_video="bouts/title_bout.mp4"):
    king = d.get("king") or "TBD"
    standings = d.get("standings", [])
    results = d.get("results", [])
    sub = d.get("substrate", "PD")
    gen = d.get("generated_utc", "")

    rows = ""
    for i, s in enumerate(standings, 1):
        medal = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(i, f"{i}")
        crown = " &#128081;" if s["name"] == king else ""
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
    <a href="index.html" class="back">← Back</a>
  </div>
  <h1>King of the <span class="red">Hill</span></h1>
  <div class="sub">Autonomous humanoid combat league &middot; MuJoCo + RL
   &middot; substrate: {html.escape(str(sub))}</div>

  <div class="king-banner">
    <div class="lbl">&#128081; Current King</div>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", default="docs/league_test.json")
    ap.add_argument("--out", default="docs/league.html")
    a = ap.parse_args()
    d = json.load(open(a.standings))
    # map (red,blue) -> mp4 path if rendered
    bout_map = {}
    for r in d.get("results", []):
        if r.get("mp4"):
            bout_map[(r["red"], r["blue"])] = r["mp4"]
    # Find the latest title bout video
    bouts_dir = os.path.join(os.path.dirname(a.out), "bouts")
    title_video = "bouts/title_bout.mp4"  # default
    if os.path.exists(bouts_dir):
        title_files = sorted([f for f in os.listdir(bouts_dir)
                              if f.startswith("title_cycle") and f.endswith(".mp4")])
        if title_files:
            title_video = f"bouts/{title_files[-1]}"  # latest cycle
        elif os.path.exists(os.path.join(bouts_dir, "title_bout.mp4")):
            title_video = "bouts/title_bout.mp4"

    html_txt = page(d, bout_map, title_video)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html_txt)
    print(f"[page] wrote {a.out} ({len(html_txt)} chars), "
          f"{len(bout_map)} bouts with video")


if __name__ == "__main__":
    main()
