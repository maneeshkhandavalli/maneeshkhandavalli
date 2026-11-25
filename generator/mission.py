"""Mission log: rotating ASCII ice giant with a lit ring + a typed terminal log (SMIL, loops)."""
import numpy as np, json, sys
from asciisvg import *

C, R, SS = 240, 34, 3
NF, FPS = 1, 8          # 1 = still planet (light). Set e.g. 64 for a spinning planet (heavier).
X, Y = grid(C, R, SS)
cx, cy, RP = 52.0, R / 2 / 0.6, 21.0
TILT = -0.14
ci, si = np.cos(TILT), np.sin(TILT)
dx0, dy0 = (X - cx) / RP, (Y - cy) / RP
dx, dy = ci * dx0 - si * dy0, si * dx0 + ci * dy0          # planet frame (axis tilted)
d2 = dx * dx + dy * dy
on = d2 < 1
nz = np.sqrt(np.clip(1 - d2, 0, 1))
lat = np.arcsin(np.clip(dy, -1, 1))
lon0 = np.arctan2(dx, nz)

# band texture (lat x lon), periodic in lon
BT = (0.9 * np.sin(np.linspace(0, 1, 512)[:, None] * 2 * np.pi * 7.5)
      + 0.55 * np.sin(np.linspace(0, 1, 512)[:, None] * 2 * np.pi * 19 + 1.3)
      + 0.45 * periodic_noise((512, 1024), (2.5, 60), 3) + 0.12 * periodic_noise((512, 1024), (1.2, 14), 4))
BT /= BT.std()

L = np.array([-0.55, -0.22, 0.81]); L /= np.linalg.norm(L)
# normal in planet frame -> rotate back to view frame for lighting
nxv = ci * dx + si * dy; nyv = -si * dx + ci * dy
diff = np.clip(nxv * L[0] + nyv * L[1] + nz * L[2], 0, 1)
light = np.clip(diff * 1.05 + 0.02, 0, 1) ** 0.9

# ring: plane through centre, normal tilted toward the viewer
incl = np.radians(11)
n_r = np.array([0, np.cos(incl), np.sin(incl)])          # in planet frame
zr = -(n_r[0] * dx + n_r[1] * dy) / n_r[2]
rr = np.sqrt(dx * dx + dy * dy + zr * zr)
RING = periodic_noise((1, 600), (0, 7), 9)[0]
rad = np.clip((rr - 1.4) / (2.1 - 1.4), 0, 0.999)
ring_d = np.where((rr > 1.4) & (rr < 2.1), np.clip(0.55 + 0.35 * RING[(rad * 599).astype(int)], 0, 1), 0)
ring_d *= np.where((rad > 0.55) & (rad < 0.6), 0.08, 1.0)           # a Cassini-style gap
pv = np.stack([ci * dx + si * dy, -si * dx + ci * dy, zr], -1)        # ring point, view frame
along = pv @ L
perp = np.linalg.norm(pv - along[..., None] * L, axis=-1)
ring_shadow = (along < 0) & (perp < 1.0)
ring_front = zr > np.where(on, nz, -9)
ring_col = np.array([1.0, 0.72, 0.38])

def frame(t):
    lon = lon0 + 2 * np.pi * t
    b = sample2d(BT, lon / (2 * np.pi), (lat / np.pi + 0.5))
    # storm, rides the surface
    sl = (np.angle(np.exp(1j * (lon - 2.2))) / 0.32) ** 2 + ((lat - 0.42) / 0.11) ** 2
    storm = np.exp(-sl * 1.6) * on
    tone = np.clip(0.5 + 0.22 * b, 0, 1)
    col = ramp_color(tone + 0.25 * storm, [0, 0.35, 0.55, 0.75, 1.1],
                     [[20, 40, 80], [45, 105, 150], [90, 175, 190], [175, 225, 225], [255, 235, 210]])
    rim = np.clip(1 - nz, 0, 1) ** 3 * 0.35 * (diff > 0.02)
    planet = col * ((0.55 + 0.45 * tone) * light + rim)[..., None] * 1.6
    planet = planet * on[..., None]
    ring = ring_col * (ring_d * (0.2 + 0.8 * (~ring_shadow)) * 0.6)[..., None]
    behind = (~ring_front) & on
    rgb = np.where(behind[..., None], planet, planet * (1 - ring_d[..., None] * 0.85) + ring)
    st = starfield(X, Y, 5, density=0.0012, t=t) * (~on) * (ring_d < 0.05)
    rgb = rgb + st[..., None] * np.array([0.8, 0.85, 1.0])
    lum, c = to_cells(rgb, R, C, SS)
    return asciify(lum, c, exposure=2.4)

frames = [frame(0.15 + f / NF) for f in range(NF)]

# ---- terminal box drawn in the grid ----
BX0, BX1, BY0, BY1 = 102, 236, 2, R - 3
dim = np.array([120, 95, 60], np.uint8)
for ch, cols in frames:
    ch[BY0:BY1 + 1, BX0:BX1 + 1] = " "
    for x in range(BX0, BX1 + 1):
        for y in (BY0, BY1):
            ch[y, x] = "-" if (x - BX0) % 2 == 0 else " "; cols[y, x] = dim
    for y in range(BY0, BY1 + 1):
        for x in (BX0, BX1):
            ch[y, x] = "|" if (y - BY0) % 1 == 0 else " "; cols[y, x] = dim
    for x, y in ((BX0, BY0), (BX1, BY0), (BX0, BY1), (BX1, BY1)):
        ch[y, x] = "+"; cols[y, x] = np.array([240, 179, 90], np.uint8)
    put_text(ch, cols, BY0, BX0 + 4, "[ mission log ]", np.array([240, 179, 90], np.uint8))
    put_text(ch, cols, BY1, BX1 - 30, "[ channel open :: 7.4 kHz ]", dim)

# ---- typed lines (SMIL, big text so it stays readable) ----
cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "profile.json"))
lines = cfg["log"]
BIG_W, BIG_H, FS = 12, 20, 19.2
x0 = (BX0 + 4) * CW; y0 = (BY0 + 3) * CH
T, CPS, HOLD_END = 16.0, 26.0, 14.6
svg, t = [], 0.6
cur_t, cur_x, cur_y = [0.0], [x0], [y0]
for i, (k, v) in enumerate(lines):
    txt = f"> {k} {'.' * max(2, 14 - len(k))} {v}"
    n = len(txt); y = y0 + i * (BIG_H + 6)
    times, vals = [0.0], [0]
    for j in range(n + 1):
        times.append(t + j / CPS); vals.append(j * BIG_W)
        cur_t.append(t + j / CPS); cur_x.append(x0 + j * BIG_W); cur_y.append(y)
    times.append(HOLD_END); vals.append(0)
    kt = ";".join(f"{a / T:.4f}" for a in times); vv = ";".join(map(str, vals))
    svg.append(f'<clipPath id="c{i}"><rect x="{x0}" y="{y}" height="{BIG_H + 4}" width="0">'
               f'<animate attributeName="width" dur="{T}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{vv}"/></rect></clipPath>')
    head = f"> {k} {'.' * max(2, 14 - len(k))} "
    svg.append(f'<text clip-path="url(#c{i})" x="{x0}" y="{y + 15}" class="big" textLength="{n * BIG_W}">'
               f'<tspan class="k">{head}</tspan><tspan class="v">{v}</tspan></text>')
    t += n / CPS + 0.45
cur_t.append(HOLD_END); cur_x.append(x0); cur_y.append(y0)
kt = ";".join(f"{a / T:.4f}" for a in cur_t)
svg.append(f'<rect class="cur" width="{BIG_W - 2}" height="{BIG_H - 4}" x="{x0}" y="{y0}">'
           f'<animate attributeName="x" dur="{T}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{";".join(map(str, cur_x))}"/>'
           f'<animate attributeName="y" dur="{T}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{";".join(str(v + 2) for v in cur_y)}"/>'
           f'<animate attributeName="opacity" dur="0.9s" repeatCount="indefinite" calcMode="discrete" values="1;0"/></rect>')
css = (f".big{{font:500 {FS}px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}"
       ".k{fill:rgb(139,148,158)}.v{fill:rgb(240,179,90)}rect.cur{fill:rgb(240,179,90)}")
open("out/mission.svg", "w").write(build(frames, FPS, extra_svg="".join(svg), extra_css=css))
print("mission.svg written")
