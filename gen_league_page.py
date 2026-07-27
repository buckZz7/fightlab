"""Generate the king-of-the-hill league page (docs/index.html).

Reads league_standings.json (+ rendered bout MP4 paths) and emits a
self-contained dark-themed page: king banner, standings table (ELO,
W/L/D), embedded bout videos.

Usage:
  python3 gen_league_page.py --standings docs/league_test.json \
      --out docs/index.html
"""
import os, sys, argparse, json, datetime, html


def page(d, bout_map):
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
<style>
  :root {{ --bg:#0a0c10; --card:#12151c; --line:#232936;
          --txt:#e6e9ef; --mut:#8b93a3; --acc:#e8443f; --acc2:#2f6fff;
          --gold:#f4c542; }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--txt);
         font:15px/1.6 system-ui,Segoe UI,Roboto,Helvetica,Arial;
         max-width:920px; margin:0 auto; padding:28px 18px 60px }}
  h1 {{ font-size:26px; letter-spacing:.5px }}
  .sub {{ color:var(--mut); margin-top:4px; font-size:13px }}
  .king-banner {{ margin:22px 0; padding:20px 22px; border-radius:14px;
    background:linear-gradient(135deg,#1a1408,#120e05);
    border:1px solid #4a3a12; }}
  .king-banner .lbl {{ color:var(--gold); font-size:12px;
    letter-spacing:2px; text-transform:uppercase }}
  .king-banner .who {{ font-size:24px; font-weight:700; margin-top:4px }}
  .card {{ background:var(--card); border:1px solid var(--line);
          border-radius:14px; padding:18px 20px; margin-top:18px }}
  h2 {{ font-size:17px; margin-bottom:12px; color:var(--txt) }}
  table {{ width:100%; border-collapse:collapse; font-variant-numeric:
          tabular-nums }}
  th, td {{ text-align:left; padding:9px 10px; border-bottom:
          1px solid var(--line) }}
  th {{ color:var(--mut); font-size:12px; text-transform:uppercase;
       letter-spacing:.5px; font-weight:600 }}
  td.elo {{ font-weight:700; color:var(--gold) }}
  td.pos {{ width:36px }}
  .king-row td {{ background:#1a1610 }}
  .bouts {{ display:grid; grid-template-columns:1fr; gap:16px }}
  .bout {{ background:var(--card); border:1px solid var(--line);
          border-radius:14px; overflow:hidden }}
  .bout-head {{ padding:12px 16px; display:flex; gap:10px;
    align-items:center; border-bottom:1px solid var(--line);
    font-weight:600 }}
  .bout-head .red {{ color:#ff7b76 }} .bout-head .blue {{ color:#7aa8ff }}
  .bout-head .vs {{ color:var(--mut); font-weight:400; font-size:12px }}
  .bout video {{ width:100%; display:block; background:#000; aspect-ratio:16/9 }}
  .bout-meta {{ padding:10px 16px; color:var(--mut); font-size:13px }}
  footer {{ margin-top:26px; color:var(--mut); font-size:12px;
          text-align:center }}
  a {{ color:var(--acc2) }}
</style>
</head>
<body>
  <h1>FightLab &#129354; King of the Hill</h1>
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

  <footer>
    Generated {html.escape(gen)} UTC &middot; FightLab &middot;
    <a href="https://github.com/buckZz7/fightlab">github.com/buckZz7/fightlab</a>
  </footer>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standings", default="docs/league_test.json")
    ap.add_argument("--out", default="docs/index.html")
    a = ap.parse_args()
    d = json.load(open(a.standings))
    # map (red,blue) -> mp4 path if rendered
    bout_map = {}
    for r in d.get("results", []):
        if r.get("mp4"):
            bout_map[(r["red"], r["blue"])] = r["mp4"]
    html_txt = page(d, bout_map)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html_txt)
    print(f"[page] wrote {a.out} ({len(html_txt)} chars), "
          f"{len(bout_map)} bouts with video")


if __name__ == "__main__":
    main()
