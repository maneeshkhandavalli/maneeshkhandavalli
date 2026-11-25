"""Tech stack orbit: boiling ASCII star, shaded planets on tilted orbits, labelled with your stack."""
import numpy as np, json, sys
from asciisvg import *

cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "profile.json"))
STACK = cfg["stack"][:6]
C, R, SS = 240, 52, 3
X, Y = grid(C, R, SS)
cx, cy = 120.0, R / 2 / 0.6 + 1.0
ELEV = np.radians(17)
SR = 8.0

# ---------------- static base: stars, nebula, orbit paths ----------------
neb = 0.0
st = starfield(X, Y, 21, density=0.0016, twinkle=False)
base_rgb = neb + st[..., None] * np.array([0.8, 0.85, 1.0])
lum, col = to_cells(base_rgb, R, C, SS)
bch, bcol = asciify(lum, col, exposure=2.2, floor=0.05)
ORBITS = [34.0, 62.0, 94.0]
for a in ORBITS:
    for th in np.linspace(0, 2 * np.pi, int(a * 7)):
        x = cx + a * np.cos(th); yy = (cy - a * np.sin(th) * np.sin(ELEV)) * 0.6
        r_, c_ = int(yy), int(x)
        if 0 <= r_ < R and 0 <= c_ < C:
            back = np.sin(th) > 0
            if np.hypot((x - cx), (yy / 0.6 - cy)) < SR * 1.3: continue
            bch[r_, c_] = "." if not back else "`"
            bcol[r_, c_] = (90, 80, 70) if back else (150, 120, 80)

# ---------------- star layer (24-frame boil) ----------------
SN = periodic_noise((256, 512), (3, 6), 5) + 0.5 * periodic_noise((256, 512), (1.2, 2.5), 6)
SN /= SN.std()
dx, dy = (X - cx) / SR, (Y - cy) / SR
rr = np.hypot(dx, dy); ang = np.arctan2(dy, dx)
disc = rr < 1
nz = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
slat = np.arcsin(np.clip(dy, -1, 1)); slon = np.arctan2(dx, nz)
STREAM = periodic_noise((1, 720), (0, 9), 8)[0]
def star_frame(t):
    n1 = sample2d(SN, slon / (2 * np.pi) + 0.06 * t, slat / np.pi + 0.5)
    n2 = sample2d(SN, slon / (2 * np.pi) + 0.06 * (t - 1), slat / np.pi + 0.5 + 0.37)
    n = (n1 * (1 - t) + n2 * t) / np.sqrt((1 - t) ** 2 + t ** 2)
    limb = 0.35 + 0.65 * nz ** 0.6
    surf = np.clip(limb * (1 + 0.28 * n), 0, None)
    scol = ramp_color(surf, [0.2, 0.55, 0.85, 1.1], [[200, 70, 20], [255, 150, 50], [255, 215, 150], [255, 248, 235]])
    s = scol * (surf * 2.2)[..., None] * disc[..., None]
    k = ((ang / (2 * np.pi)) % 1 * 720).astype(int)
    k2 = (k + int(t * 720 / 6)) % 720                       # streamers drift a 1/6 turn per loop..
    strm = STREAM[k] * (1 - t) + STREAM[(k + 120) % 720] * t   # ..and cross-fade so it loops
    cor = np.exp(-(rr - 1) * 2.6) * (0.55 + 0.35 * strm) * (~disc) * (rr < 2.6)
    s = s + np.array([1.0, 0.55, 0.2]) * cor[..., None] * 0.9
    lum, c = to_cells(s, R, C, SS)
    ch, cc = asciify(lum, c, exposure=2.4, floor=0.04)
    rcell = np.hypot((np.arange(C)[None] + 0.5 - cx) / SR, ((np.arange(R)[:, None] + 0.5) / 0.6 - cy) / SR)
    ch = np.where((rcell < 2.5) & ((ch != " ") | (rcell < 1.05)), ch, TRANSPARENT)
    return ch, cc
star_frames = [star_frame(f / 24) for f in range(24)]

# ---------------- planets layer ----------------
NF, FPS = 200, 10
PAL = [[(30, 60, 140), (110, 170, 240)], [(20, 90, 80), (120, 220, 190)], [(120, 50, 30), (250, 170, 110)],
       [(70, 40, 120), (190, 160, 245)], [(110, 90, 40), (240, 220, 150)], [(120, 30, 70), (240, 140, 180)]]
BANDS = [periodic_noise((128, 256), (1.5, 14), 30 + i) * 0.6 + np.sin(np.linspace(0, 1, 128)[:, None] * np.pi * (5 + i)) for i in range(6)]
planets = []
for i, name in enumerate(STACK):
    orb = i // 2
    planets.append(dict(name=name, a=ORBITS[orb], rp=[5.0, 6.5, 7.5][orb], rev=[3, 2, 1][orb],
                        ph=(i % 2) * np.pi + orb * 1.1, pal=PAL[i], tex=BANDS[i], ring=(i == 4)))

def planet_frame(t):
    rgb = np.zeros(X.shape + (3,)); hit = np.zeros(X.shape, bool); zbuf = np.full(X.shape, -1e9)
    labels = []
    for p in planets:
        th = p["ph"] + 2 * np.pi * p["rev"] * t
        depth = -np.sin(th)                                   # + toward camera
        sc = 1 + 0.12 * depth
        px = cx + p["a"] * np.cos(th); py = cy - p["a"] * np.sin(th) * np.sin(ELEV)
        rp = p["rp"] * sc
        ddx, ddy = (X - px) / rp, (Y - py) / rp
        d2 = ddx ** 2 + ddy ** 2
        on = d2 < 1
        behind_star = (depth < 0) & (np.hypot(X - cx, Y - cy) < SR * 1.02)
        on &= ~behind_star
        pz = np.sqrt(np.clip(1 - d2, 0, 1))
        Lv = np.array([-np.cos(th), np.sin(th) * np.sin(ELEV), np.sin(th) * np.cos(ELEV)])
        Lv /= np.linalg.norm(Lv)
        dif = np.clip(ddx * Lv[0] + ddy * Lv[1] + pz * Lv[2], 0, 1)
        lon = np.arctan2(ddx, pz) / (2 * np.pi) + 3 * t * (1 if p["rev"] != 2 else -1)
        tex = sample2d(p["tex"], lon, np.arcsin(np.clip(ddy, -1, 1)) / np.pi + 0.5)
        tone = np.clip(0.5 + 0.25 * tex, 0, 1)
        lo, hi = np.array(p["pal"][0]) / 255, np.array(p["pal"][1]) / 255
        c = lo + (hi - lo) * tone[..., None]
        shade = (0.1 + 2.4 * dif ** 0.8)
        if p["ring"]:
            zr_ok = np.abs(ddy) < 0.22
            rr2 = np.hypot(ddx, ddy / 0.2)
            ring = (rr2 > 1.35) & (rr2 < 2.0) & ~((ddy < 0) & on) & ~behind_star
            rgb = np.where(ring[..., None] & ~hit[..., None], np.array([0.85, 0.7, 0.45]) * 0.9, rgb)
            hit |= ring
        rgb = np.where(on[..., None], c * shade[..., None], rgb)
        hit |= on
        if not ((depth < 0) and abs(px - cx) < SR * 1.3):
            labels.append((px, py - rp - 3.2, p["name"], depth))
    lum, cc = to_cells(rgb, R, C, SS)
    ch, cc = asciify(lum, cc, exposure=2.6, floor=0.0)
    hc = hit.reshape(R, SS, C, SS).any(axis=(1, 3))
    ch = np.where(hc, ch, TRANSPARENT)
    ex = "".join(f'<text class="lbl{" dim" if d < 0 else ""}" x="{x * CW:.0f}" y="{y * 0.6 * CH:.0f}" text-anchor="middle">{n}</text>'
                 for x, y, n, d in labels)
    return (ch, cc), ex

pf = [planet_frame(f / NF) for f in range(NF)]
svg = build_layers((bch, bcol), [dict(frames=star_frames, fps=8), dict(frames=[a for a, _ in pf], fps=FPS, extra=[b for _, b in pf])],
                   extra_svg=f'<text class="hdr" x="{4 * CW}" y="{2 * CH + 12}">[ systems onboard ]</text>',
                   extra_css=".lbl{font:500 17px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(230,237,243)}"
                             ".dim{fill:rgb(139,148,158)}"
                             ".hdr{font:500 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(240,179,90)}")
open("out/orbit.svg", "w").write(svg)
print("orbit.svg written")
