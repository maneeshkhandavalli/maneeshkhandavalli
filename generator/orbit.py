"""Tech stack orbit, lightweight version.

Everything is drawn ONCE as ASCII (background, star, each planet as a small sprite).
Planets then glide along their orbits with SMIL <animateMotion>, which browsers move
cheaply (no text re-layout per frame). Each planet exists twice: one copy under the
star (seen when it passes behind) and one copy over it that only shows on the near half.
"""
import numpy as np, json, sys
from asciisvg import *
from asciisvg import _quantize, _row_text

cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "profile.json"))
STACK = cfg["stack"][:6]
C, R, SS = 240, 52, 3
X, Y = grid(C, R, SS)
cx, cy = 120.0, R / 2 / 0.6 + 1.0          # centre (cols, Y-units)
ELEV = np.radians(17)
SR = 8.0
ORBITS = [34.0, 62.0, 94.0]
PERIOD = [24.0, 40.0, 64.0]                 # seconds per revolution (inner .. outer)

# ---------- static background: stars + dotted orbit paths ----------
st = starfield(X, Y, 21, density=0.0016, twinkle=False)
lum, col = to_cells(st[..., None] * np.array([0.8, 0.85, 1.0]), R, C, SS)
bch, bcol = asciify(lum, col, exposure=2.2, floor=0.05)
for a in ORBITS:
    for th in np.linspace(0, 2 * np.pi, int(a * 7)):
        x = cx + a * np.cos(th); yy = (cy - a * np.sin(th) * np.sin(ELEV)) * 0.6
        r_, c_ = int(yy), int(x)
        if 0 <= r_ < R and 0 <= c_ < C:
            back = np.sin(th) > 0
            bch[r_, c_] = "`" if back else "."
            bcol[r_, c_] = (90, 80, 70) if back else (150, 120, 80)

# ---------- the star (static ASCII, drawn over the back halves of the orbits) ----------
SN = periodic_noise((256, 512), (3, 6), 5) + 0.5 * periodic_noise((256, 512), (1.2, 2.5), 6)
SN /= SN.std()
dx, dy = (X - cx) / SR, (Y - cy) / SR
rr = np.hypot(dx, dy); ang = np.arctan2(dy, dx)
disc = rr < 1
nz = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
n = sample2d(SN, np.arctan2(dx, nz) / (2 * np.pi), np.arcsin(np.clip(dy, -1, 1)) / np.pi + 0.5)
surf = np.clip((0.35 + 0.65 * nz ** 0.6) * (1 + 0.28 * n), 0, None)
scol = ramp_color(surf, [0.2, 0.55, 0.85, 1.1], [[200, 70, 20], [255, 150, 50], [255, 215, 150], [255, 248, 235]])
s = scol * (surf * 2.2)[..., None] * disc[..., None]
STREAM = periodic_noise((1, 720), (0, 9), 8)[0]
strm = STREAM[((ang / (2 * np.pi)) % 1 * 720).astype(int)]
cor = np.exp(-(rr - 1) * 2.6) * (0.55 + 0.35 * strm) * (~disc) * (rr < 2.6)
s = s + np.array([1.0, 0.55, 0.2]) * cor[..., None] * 0.9
lum, c = to_cells(s, R, C, SS)
sch, scol_ = asciify(lum, c, exposure=2.4, floor=0.04)
rcell = np.hypot((np.arange(C)[None] + 0.5 - cx) / SR, ((np.arange(R)[:, None] + 0.5) / 0.6 - cy) / SR)
star_on = (rcell < 2.5) & (sch != " ")

# ---------- planet sprites ----------
PAL = [[(30, 60, 140), (110, 170, 240)], [(20, 90, 80), (120, 220, 190)], [(120, 50, 30), (250, 170, 110)],
       [(70, 40, 120), (190, 160, 245)], [(110, 90, 40), (240, 220, 150)], [(120, 30, 70), (240, 140, 180)]]
L = np.array([-0.55, -0.35, 0.76]); L /= np.linalg.norm(L)

def sprite(i, rp, ring):
    w = int(rp * (4.6 if ring else 2.4)) | 1; h = int(rp * 1.5 * 0.6 * (1.25 if ring else 1)) + 3
    xs, ys = grid(w, h, SS)
    ddx = (xs - w / 2) / rp; ddy = (ys - h / 2 / 0.6) / rp
    d2 = ddx ** 2 + ddy ** 2; on = d2 < 1
    pz = np.sqrt(np.clip(1 - d2, 0, 1))
    dif = np.clip(ddx * L[0] + ddy * L[1] + pz * L[2], 0, 1)
    tex = periodic_noise((128, 256), (1.5, 14), 30 + i) * 0.6 + np.sin(np.linspace(0, 1, 128)[:, None] * np.pi * (5 + i))
    t = sample2d(tex, np.arctan2(ddx, pz) / (2 * np.pi), np.arcsin(np.clip(ddy, -1, 1)) / np.pi + 0.5)
    tone = np.clip(0.5 + 0.25 * t, 0, 1)
    lo, hi = np.array(PAL[i][0]) / 255, np.array(PAL[i][1]) / 255
    rgb = (lo + (hi - lo) * tone[..., None]) * (0.1 + 2.4 * dif ** 0.8)[..., None] * on[..., None]
    if ring:
        rr2 = np.hypot(ddx, ddy / 0.2)
        rg = (rr2 > 1.35) & (rr2 < 2.0) & ~((ddy < 0) & on)
        rgb = np.where(rg[..., None], np.array([0.85, 0.7, 0.45]) * 0.9, rgb)
    lum, cc = to_cells(rgb, h, w, SS)
    ch, cc = asciify(lum, cc, exposure=2.6, floor=0.0)
    return ch, cc

planets = []
for i, name in enumerate(STACK):
    orb = i // 2
    rp = [5.0, 6.5, 7.5][orb]
    ch, cc = sprite(i, rp, ring=(i == 4))
    planets.append(dict(name=name, orb=orb, rp=rp, ch=ch, cc=cc, ph=(i % 2) * 0.5 + orb * 0.17))

# ---------- shared colour classes, then emit ----------
cent, q = _quantize([(bch, bcol), (np.where(star_on, sch, " "), scol_)] + [(p["ch"], p["cc"]) for p in planets], 18)
base = "".join(_row_text(y, 0, q[0][0][y], q[0][1][y]) for y in range(R))
sk, sq = q[1]
star_rows = []
for y in range(R):
    xs = np.nonzero(star_on[y])[0]
    if len(xs): star_rows.append(_row_text(y, xs[0], sk[y, xs[0]:xs[-1] + 1], sq[y, xs[0]:xs[-1] + 1]))
star_svg = (f'<ellipse cx="{cx * CW}" cy="{cy * 0.6 * CH}" rx="{SR * 1.05 * CW}" ry="{SR * 1.05 * 0.6 * CH}"/>'
            f'<g class="cor">{"".join(star_rows)}</g>')

def path(a):
    rx, ry = a * CW, a * np.sin(ELEV) * 0.6 * CH
    x0, y0 = cx * CW, cy * 0.6 * CH
    return f"M{x0 + rx:.1f},{y0:.1f} A{rx:.1f},{ry:.1f} 0 1 0 {x0 - rx:.1f},{y0:.1f} A{rx:.1f},{ry:.1f} 0 1 0 {x0 + rx:.1f},{y0:.1f}"

under, over = [], []
for j, p in enumerate(planets):
    h, w = p["ch"].shape
    k_, qq = q[2 + j]
    rows = "".join(_row_text(y, 0, k_[y], qq[y]) for y in range(h))
    body = (f'<ellipse rx="{p["rp"] * CW * 1.02:.1f}" ry="{p["rp"] * 0.6 * CH * 1.02:.1f}"/>'
            f'<g transform="translate({-w * CW / 2:.1f},{-h * CH / 2:.1f})">{rows}</g>'
            f'<text class="lbl" y="{-p["rp"] * 0.6 * CH - 14:.0f}" text-anchor="middle">{p["name"]}</text>')
    dur = PERIOD[p["orb"]]; begin = -p["ph"] * dur
    motion = f'<animateMotion dur="{dur}s" begin="{begin:.2f}s" repeatCount="indefinite" path="{path(ORBITS[p["orb"]])}"/>'
    under.append(f'<g>{body}{motion}</g>')
    # near-half copy: visible only for the second half of each lap (the side facing us)
    over.append(f'<g opacity="0">{body}{motion}'
                f'<animate attributeName="opacity" dur="{dur}s" begin="{begin:.2f}s" repeatCount="indefinite" calcMode="discrete" keyTimes="0;0.5" values="0;1"/></g>')

css_cols = "".join(f".q{i}{{fill:rgb({c[0]},{c[1]},{c[2]})}}" for i, c in enumerate(cent))
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C*CW} {R*CH}" width="{C*CW}" height="{R*CH}" xml:space="preserve">
<style>
text{{font:9.6px ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace;white-space:pre}}
rect,ellipse{{fill:rgb{BG}}}
.lbl{{font:500 17px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(230,237,243)}}
.hdr{{font:500 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(240,179,90)}}
.cor{{animation:p 4s ease-in-out infinite}}
@keyframes p{{50%{{opacity:.82}}}}
{css_cols}
</style>
<rect width="100%" height="100%" rx="10"/>
<g>{base}</g>
{"".join(under)}
{star_svg}
{"".join(over)}
<text class="hdr" x="{4 * CW}" y="{2 * CH + 12}">[ systems onboard ]</text>
</svg>'''
open("out/orbit.svg", "w").write(svg.replace('textLength="', 'lengthAdjust="spacingAndGlyphs" textLength="'))
print("orbit.svg written")
