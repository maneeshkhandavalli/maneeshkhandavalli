"""Render the loop and export: PNG frames -> MP4 / GIF / animated WebP, and an animated SVG."""
import numpy as np, sys, os, subprocess
from PIL import Image, ImageDraw, ImageFont
import render as RD

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 72
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
C, R = RD.C, RD.R
BG = (13, 17, 23)
os.makedirs("frames", exist_ok=True); os.makedirs("out", exist_ok=True)

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
CW, CH = 6, 10

# fixed colour classes for the SVG (keeps file small): sample palette x brightness
NCOL = int(os.environ.get("NCOL", 14))
CENT = None
def quantize(c):
    d = ((c.reshape(-1, 1, 3).astype(float) - CENT[None]) ** 2).sum(-1)
    return CENT[d.argmin(1)].reshape(c.shape).astype(int)

frames = []
for f in range(NF):
    chars, col, L = RD.frame(f / NF)
    chars, col = RD.add_comets(chars, col, f / NF)
    frames.append((chars, col))
    im = Image.new("RGB", (C * CW, R * CH), BG); d = ImageDraw.Draw(im)
    for y in range(R):
        for x in range(C):
            ch = chars[y, x]
            if ch != " ":
                d.text((x * CW, y * CH - 1), ch, font=font, fill=tuple(int(v) for v in col[y, x]))
    im.save(f"frames/f{f:03d}.png")
    print("frame", f, flush=True) if f % 12 == 0 else None

from scipy.cluster.vq import kmeans2
allc = np.concatenate([col[chars != " "] for chars, col in frames[::6]]).astype(float)
rs = np.random.default_rng(0); allc = allc[rs.choice(len(allc), min(40000, len(allc)), replace=False)]
CENT, _ = kmeans2(allc, NCOL, seed=1, minit="++")
CENT = np.clip(CENT, 0, 255).round()

# ---------------- video / raster exports ----------------
def ff(*a): subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)
ff("-framerate", str(FPS), "-i", "frames/f%03d.png", "-c:v", "libx264", "-preset", "veryfast",
   "-crf", "18", "-pix_fmt", "yuv420p", "out/blackhole.mp4")
ims = [Image.open(f"frames/f{i:03d}.png") for i in range(NF)]
ims[0].save("out/blackhole.webp", save_all=True, append_images=ims[1:], duration=int(1000 / FPS),
            loop=0, quality=70, method=6)
ff("-framerate", str(FPS), "-i", "frames/f%03d.png", "-vf",
   "split[a][b];[a]palettegen=max_colors=32:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
   "-loop", "0", "out/blackhole.gif")

# ---------------- animated SVG ----------------
esc = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
palette = {}
def cls(c):
    k = tuple(int(v) for v in c)
    if k not in palette: palette[k] = f"c{len(palette):x}"
    return palette[k]

dur = NF / FPS
body = []
for i, (chars, col) in enumerate(frames):
    q = quantize(col)
    rows = []
    for y in range(R):
        line = chars[y]
        if not "".join(line).strip(): continue
        parts, cur, buf = [], None, ""
        for x in range(C):
            ch = line[x]
            k = None if ch == " " else cls(q[y, x])
            if ch == " " or k == cur:
                buf += esc.get(ch, ch)
            else:
                if buf: parts.append(f'<tspan class="{cur}">{buf}</tspan>' if cur else buf)
                cur, buf = k, esc.get(ch, ch)
        if buf.strip(): parts.append(f'<tspan class="{cur}">{buf}</tspan>' if cur else buf)
        rows.append(f'<text y="{y * CH + 8}" textLength="{C * CW}">{"".join(parts)}</text>')
    body.append(f'<g class="f" style="animation-delay:{i / FPS:.3f}s">{"".join(rows)}</g>')

css_cols = "".join(f".{v}{{fill:rgb({k[0]},{k[1]},{k[2]})}}" for k, v in palette.items())
pct = 100 / NF
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C*CW} {R*CH}" width="{C*CW}" height="{R*CH}" xml:space="preserve">
<style>
text{{font:9.6px ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace;white-space:pre}}
text{{lengthAdjust:spacingAndGlyphs}}
.f{{visibility:hidden;animation:k {dur:.3f}s step-end infinite}}
@keyframes k{{0%{{visibility:visible}}{pct:.4f}%{{visibility:hidden}}}}
{css_cols}
</style>
<rect width="100%" height="100%" rx="10" fill="rgb{BG}"/>
{"".join(body)}
</svg>'''
svg = svg.replace('textLength="', 'lengthAdjust="spacingAndGlyphs" textLength="')
open("out/blackhole.svg", "w").write(svg)
for fn in sorted(os.listdir("out")):
    print(fn, f"{os.path.getsize('out/'+fn)/1e6:.2f} MB")
print("svg colours:", len(palette))
