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
LINE = np.array([150, 105, 60], np.uint8)
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
    if p["lv"] == 0: put(bch, bcol, p["y"], p["x"], ".", np.array([75, 82, 95], np.uint8))

# ---- star sprites per level, with twinkle variants ----
COLS = {1: (230, 140, 60), 2: (255, 190, 110), 3: (255, 225, 175), 4: (235, 242, 255)}
SPR = {1: [{(0, 0): "+"}, {(0, 0): "."}],
       2: [{(0, 0): "*"}, {(0, 0): "+"}],
       3: [{(0, -1): "-", (0, 0): "*", (0, 1): "-"}, {(0, 0): "*"}, {(0, -1): "-", (0, 0): "+", (0, 1): "-"}],
       4: [{(0, -2): "-", (0, -1): "-", (0, 0): "@", (0, 1): "-", (0, 2): "-", (-1, 0): "|", (1, 0): "|"},
           {(0, -1): "-", (0, 0): "*", (0, 1): "-", (-1, 0): "'", (1, 0): "."},
           {(0, -3): "-", (0, -2): "-", (0, -1): "=", (0, 0): "@", (0, 1): "=", (0, 2): "-", (0, 3): "-", (-1, 0): "|", (1, 0): "|"}]}

def draw_stars(ch, cl, t):
    for p in act:
        v = SPR[p["lv"]]
        k = int(((p["cyc"] * t + p["ph"]) % 1) * len(v)) if t is not None else 0
        c = np.array(COLS[p["lv"]], float)
        for (dy, dx), s in v[k].items():
            fade = 1.0 if (dy, dx) == (0, 0) else 0.6
            put(ch, cl, p["y"] + dy, p["x"] + dx, s, (c * fade).astype(np.uint8))

base_ch, base_cl = bch.copy(), bcol.copy()
draw_stars(base_ch, base_cl, None)
NF, FPS = 0, 6           # 0 = still image (light). Set 24 to make the stars twinkle.
frames = []
for f in range(NF):
    ch = np.full((R, C), TRANSPARENT); cl = np.zeros((R, C, 3), np.uint8)
    # restore the background under the biggest sprite footprint, so shrinking sprites look right
    for p in act:
        for (dy, dx) in set(k for v in SPR[p["lv"]] for k in v):
            yy, xx = p["y"] + dy, p["x"] + dx
            if 0 <= yy < R and 0 <= xx < C:
                ch[yy, xx] = bch[yy, xx]; cl[yy, xx] = bcol[yy, xx]
    draw_stars(ch, cl, f / NF)
    frames.append((ch, cl))

months = []
for i, d in enumerate(days):
    dd = dt.date.fromisoformat(d["date"])
    if dd.day == 1:
        months.append((10 + ((i + off) // 7) * (220 / 52), dd.strftime("%b").lower()))
mx = "".join(f'<text class="mo" x="{x * CW:.0f}" y="{(R - 1) * CH}">{m}</text>' for x, m in months if x < 228)
hdr = f'<text class="hdr" x="{4 * CW}" y="{2 * CH + 12}">[ {S["total"]} contributions in the last year :: charted as stars ]</text>'
svg = build_layers((base_ch, base_cl), [dict(frames=frames, fps=FPS)] if NF else [], extra_svg=hdr + mx,
                   extra_css=".hdr{font:500 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(240,179,90)}"
                             ".mo{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(110,118,129)}")
open("out/constellation.svg", "w").write(svg)
print("constellation.svg written")
