"""Contribution constellation: every day of the last year is a star; busier days burn brighter and twinkle."""
import numpy as np, json, datetime as dt
from asciisvg import *

S = json.load(open("data/stats.json"))
days = S["days"]
C, R, SS = 240, 40, 3
X, Y = grid(C, R, SS)

d0 = dt.date.fromisoformat(days[0]["date"])
off = (d0.weekday() + 1) % 7                      # GitHub weeks start on Sunday
rng = np.random.default_rng(12)
pts = []
for i, d in enumerate(days):
    w, wd = divmod(i + off, 7)
    x = int(round(10 + w * (220 / 52) + rng.uniform(-1, 1)))
    y = int(round(6 + wd * 4.3 + rng.uniform(-0.6, 0.6)))
    pts.append(dict(x=x, y=y, c=d["count"], ph=rng.uniform(0, 1), cyc=int(rng.integers(1, 3)), i=i))
nz = sorted(p["c"] for p in pts if p["c"] > 0) or [1]
q = np.quantile(nz, [0.35, 0.7, 0.92])
for p in pts:
    p["lv"] = 0 if p["c"] == 0 else 1 + int(np.searchsorted(q, p["c"], side="right"))   # 1..4
    p["lv"] = min(p["lv"], 4)

# ---- background: milky-way dust (sparse, soft chars) + faint field stars ----
band = np.exp(-((Y * 0.6 - (4 + X * 0.13)) / 8.0) ** 2)
neb = nebula(X, Y, 8, scale=22)
dust_rgb = (neb * 0.25 + band[..., None] * (neb * 0.7 + np.array([0.06, 0.05, 0.09])))
dust_rgb += starfield(X, Y, 33, density=0.001, twinkle=False)[..., None] * 0.5
lum, col = to_cells(dust_rgb, R, C, SS)
bch, bcol = asciify(lum, col, exposure=2.6, floor=0.05, ramp=" .`',:;~+")
keep = np.random.default_rng(3).random((R, C)) < np.clip((1 - np.exp(-2.6 * lum)) * 3.2, 0, 1)
bch = np.where(keep, bch, " ")
bcol = (bcol * 0.75).astype(np.uint8)

def put(ch, cl, y, x, s, rgb):
    if 0 <= y < R and 0 <= x < C and s != " ":
        ch[y, x] = s; cl[y, x] = rgb

# ---- constellation lines between consecutive active days ----
act = [p for p in pts if p["lv"] > 0]
LINE = np.array([205, 145, 75], np.uint8)
bright = [p for p in act if p["lv"] >= 3]
for a, b in zip(bright, bright[1:]):
    if b["i"] - a["i"] > 24 or abs(b["x"] - a["x"]) > 22: continue
    n = max(abs(b["x"] - a["x"]), abs(b["y"] - a["y"]))
    if n < 3: continue
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    s = "-" if abs(dy) * 3 < abs(dx) else "|" if abs(dx) * 3 < abs(dy) else ("\\" if dx * dy > 0 else "/")
    for k in range(2, n - 1, 2 if s == "-" else 1):
        put(bch, bcol, round(a["y"] + dy * k / n), round(a["x"] + dx * k / n), s, LINE)
for p in pts:                                      # quiet days: dim specks
    if p["lv"] == 0: put(bch, bcol, p["y"], p["x"], ".", np.array([105, 115, 135], np.uint8))

# ---- star sprites per level (drawn as their own text so they can twinkle cheaply) ----
COLS = {1: (255, 170, 80), 2: (255, 205, 130), 3: (255, 235, 195), 4: (255, 255, 255)}
SPR = {1: ["*"],
       2: ["-*-"],
       3: [" | ", "-@-", " | "],
       4: ["  |  ", "\\ | /", "--@--", "/ | \\", "  |  "]}

def hexc(c, f=1.0): return "#%02x%02x%02x" % tuple(int(min(255, v * f)) for v in c)
groups = {0: [], 1: [], 2: []}
for p in act:
    rows = SPR[p["lv"]]
    h, w = len(rows), len(rows[0])
    g = []
    for r, line in enumerate(rows):
        y = p["y"] - h // 2 + r
        if not (0 <= y < R): continue
        x0 = p["x"] - w // 2
        spans = "".join(
            ch if ch == " " else
            f'<tspan fill="{hexc(COLS[p["lv"]], 1.0 if (r == h // 2 and k == w // 2) else 0.62)}">{ESC.get(ch, ch)}</tspan>'
            for k, ch in enumerate(line))
        g.append(f'<text x="{x0 * CW}" y="{y * CH + 8}" textLength="{w * CW}">{spans}</text>')
    groups[int(p["ph"] * 3) % 3 if p["lv"] >= 2 else 0].append("".join(g))
# quiet halo so the brightest stars read at a glance
stars_svg = "".join(f'<g class="tw{k}">{"".join(v)}</g>' for k, v in groups.items())
base_ch, base_cl = bch, bcol

months = []
for i, d in enumerate(days):
    dd = dt.date.fromisoformat(d["date"])
    if dd.day == 1:
        months.append((10 + ((i + off) // 7) * (220 / 52), dd.strftime("%b").lower()))
mx = "".join(f'<text class="mo" x="{x * CW:.0f}" y="{(R - 1) * CH}">{m}</text>' for x, m in months if x < 228)
hdr = f'<text class="hdr" x="{4 * CW}" y="{2 * CH + 12}">[ {S["total"]} contributions in the last year :: charted as stars ]</text>'
svg = build_layers((base_ch, base_cl), [], extra_svg=stars_svg + hdr + mx,
                   extra_css=".hdr{font:500 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(240,179,90)}"
                             ".mo{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(110,118,129)}"
                             ".tw1{animation:tw 3.2s ease-in-out infinite}.tw2{animation:tw 4.1s ease-in-out -1.7s infinite}"
                             "@keyframes tw{50%{opacity:.45}}")
open("out/constellation.svg", "w").write(svg)
print("constellation.svg written")
