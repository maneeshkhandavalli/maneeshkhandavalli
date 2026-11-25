"""Telemetry: big dense-ASCII numbers (rasterised from a font), a slow scan sweep, twinkling stars."""
import numpy as np, json, os
from PIL import Image, ImageDraw, ImageFont
from asciisvg import *

S = json.load(open("data/stats.json"))
STATS = [(S["total"], "contributions", "last 12 months"),
         (S["streak"], "day streak", f"best {S['best_streak']}"),
         (S["month_commits"], "commits", S["month"]),
         (S["repos"], "public repos", f"{S['stars']} star" + ("" if S["stars"] == 1 else "s"))]
C, R, SS = 240, 30, 3
X, Y = grid(C, R, SS)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "DejaVuSans-Bold.ttf")
PW = 60                                           # panel width in cols

# glyph mask on the supersampled grid
mask = np.zeros(X.shape)
H_S, W_S = X.shape
img = Image.new("L", (W_S, H_S), 0); d = ImageDraw.Draw(img)
ROW0, NUMH = 5, 13                                # number top row and height (rows)
px_h = int(NUMH * SS)
font = ImageFont.truetype(FONT, int(px_h * 1.18))
for i, (v, _, _) in enumerate(STATS):
    txt = f"{v:,}"
    # sample grid is anisotropic (cells are 0.6 wide/tall), so draw squashed then stretch
    tmp = Image.new("L", (int(PW * SS * 0.6), px_h + 10), 0)
    dt_ = ImageDraw.Draw(tmp)
    bb = dt_.textbbox((0, 0), txt, font=font)
    dt_.text(((tmp.width - (bb[2] - bb[0])) // 2 - bb[0], (tmp.height - (bb[3] - bb[1])) // 2 - bb[1]), txt, font=font, fill=255)
    tmp = tmp.resize((PW * SS, tmp.height), Image.LANCZOS)
    img.paste(tmp, (i * PW * SS, ROW0 * SS - 5))
mask = np.asarray(img, float) / 255
from scipy.ndimage import gaussian_filter
bev = gaussian_filter(mask, 2.0)
gy, gx = np.gradient(bev)
shade = np.clip(1.0 - 1.5 * (gx * 0.6 + gy * 0.8), 0.7, 1.25)
vert = np.clip((Y * 0.6 - ROW0) / NUMH, 0, 1)
num_col = ramp_color(vert, [0, 0.5, 1], [[255, 236, 205], [245, 175, 85], [205, 95, 40]])
glow = gaussian_filter(mask, 6) * 0.25

NF, FPS = 1, 10          # 1 = still image (light). Set 40 for the scan sweep.
def frame(t):
    sweep_x = -20 + (C + 40) * t
    sw = np.exp(-((X - sweep_x - (Y * 0.6) * 0.6) / 5.0) ** 2)
    num = num_col * (mask * shade * (1 + 0.9 * sw))[..., None] * 1.4
    num += np.array([1.0, 0.6, 0.25]) * glow[..., None]
    st = starfield(X, Y, 44, density=0.0011, t=t)[..., None] * np.array([0.8, 0.85, 1.0]) * (mask < 0.05)[..., None]
    lum, c = to_cells(num + st, R, C, SS)
    return asciify(lum, c, exposure=4.0, floor=0.05, gamma=0.6, min_bright=0.6, ramp=" .:-=+*#%@")

frames = [frame(f / NF) for f in range(NF)]
dim = np.array([95, 85, 70], np.uint8)
for ch, cl in frames:                              # panel brackets
    for i in range(4):
        x0, x1 = i * PW + 2, (i + 1) * PW - 3
        for (y, x, s) in ((2, x0, "+"), (2, x1, "+"), (R - 2, x0, "+"), (R - 2, x1, "+")):
            ch[y, x] = s; cl[y, x] = dim
        for x in (x0 + 1, x0 + 2, x1 - 2, x1 - 1):
            for y in (2, R - 2): ch[y, x] = "-"; cl[y, x] = dim
        for y in (3, R - 3):
            for x in (x0, x1): ch[y, x] = "|"; cl[y, x] = dim
labels = "".join(
    f'<text class="lb" x="{(i * PW + PW / 2) * CW:.0f}" y="{(ROW0 + NUMH + 3.2) * CH:.0f}" text-anchor="middle">{a}</text>'
    f'<text class="sb" x="{(i * PW + PW / 2) * CW:.0f}" y="{(ROW0 + NUMH + 5.4) * CH:.0f}" text-anchor="middle">{b}</text>'
    for i, (_, a, b) in enumerate(STATS))
hdr = f'<text class="hdr" x="{4 * CW}" y="{1 * CH + 9}">[ telemetry :: auto-updated daily ]</text>'
svg = build(frames, FPS, extra_svg=hdr + labels,
            extra_css=".lb{font:500 17px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(230,237,243)}"
                      ".sb{font:14px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(139,148,158)}"
                      ".hdr{font:500 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:rgb(240,179,90)}")
open("out/telemetry.svg", "w").write(svg)
print("telemetry.svg written")
