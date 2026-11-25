"""Shade traced geometry into ASCII frames (chars + colors) for a seamless loop."""
import numpy as np
from scipy.ndimage import gaussian_filter

G = np.load("geometry.npz")
C, R, SS = int(G["C"]), int(G["R"]), int(G["SS"])
R_IN, R_OUT = float(G["R_IN"]), float(G["R_OUT"])
hit_r, hit_phi, hit_g, nhit = G["hit_r"], G["hit_phi"], G["hit_g"], G["nhit"]
escaped, edir = G["escaped"], G["edir"]
N = hit_r.shape[1]

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
RAMP_ARR = np.array(list(RAMP))

# ---------- disk texture: periodic in phi, streaky along the orbit ----------
rng = np.random.default_rng(7)
NR, NP = 320, 4096
def layer(sr, sp, w):
    n = gaussian_filter(rng.standard_normal((NR, NP)), (sr, sp), mode=("nearest", "wrap"))
    return w * n / n.std()
TEX = layer(0.7, 90, 0.55) + layer(0.5, 18, 0.45) + layer(3.0, 220, 0.6) + layer(0.35, 5, 0.25)
TEX /= TEX.std()

def tex_at(r, phi):
    ri = np.clip((r - R_IN) / (R_OUT - R_IN) * (NR - 1), 0, NR - 1)
    pi = (phi / (2 * np.pi) % 1.0) * NP
    r0 = ri.astype(int); r1 = np.minimum(r0 + 1, NR - 1); fr = ri - r0
    p0 = pi.astype(int) % NP; p1 = (p0 + 1) % NP; fp = pi - np.floor(pi)
    a = TEX[r0, p0] * (1 - fp) + TEX[r0, p1] * fp
    b = TEX[r1, p0] * (1 - fp) + TEX[r1, p1] * fp
    return a * (1 - fr) + b * fr

OMEGA0 = 2 * np.pi * 0.45          # inner-edge turns per loop (slow)
def omega(r): return OMEGA0 * (R_IN / r) ** 1.5

def emission(r):
    x = R_IN / r
    return x ** 3 * np.clip(1 - np.sqrt(x * 0.985), 0, None) / 0.0547

def palette(temp):
    stops = np.array([0.0, 0.35, 0.6, 0.85, 1.15, 1.6])
    cols = np.array([[90, 22, 6], [200, 70, 18], [255, 140, 50],
                     [255, 205, 140], [255, 240, 220], [215, 228, 255]], float)
    t = np.clip(temp, 0, 1.6)
    out = np.empty(t.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(t, stops, cols[:, c])
    return out / 255

# ---------- starfield (static, lensed) ----------
th = np.arccos(np.clip(edir[:, 1], -1, 1)); ph = np.arctan2(edir[:, 2], edir[:, 0])
SG = 3200
cell = (np.floor(th / np.pi * SG).astype(np.int64) * 3 * SG + np.floor((ph + np.pi) / (2 * np.pi) * 2 * SG).astype(np.int64))
h = (cell * 2654435761) % 4294967296 / 4294967296.0
star = np.where(h > 0.9993, ((h - 0.9993) / 0.0007) ** 3 * 0.9 + 0.05, 0.0) * escaped

def shade(t):
    """t in [0,1) -> per-sample rgb, luminance."""
    rgb = np.zeros((N, 3)); acc = np.zeros(N)
    for k in range(3):
        m = nhit > k
        r, phi, g = hit_r[k, m], hit_phi[k, m], hit_g[k, m]
        # seamless loop: blend two time-shifted textures, variance preserved
        n1 = tex_at(r, phi - omega(r) * t)
        n2 = tex_at(r, phi - omega(r) * (t - 1.0))
        n = (n1 * (1 - t) + n2 * t) / np.sqrt((1 - t) ** 2 + t ** 2)
        e = emission(r)
        I = e * g ** 2.3 * np.clip(1 + 0.75 * n, 0.05, None)
        fade = np.clip((R_OUT - r) / 4.0, 0, 1) ** 1.5 * np.clip((r - R_IN) / 0.25, 0, 1)
        alpha = np.clip(0.55 + 0.4 * fade * np.clip(1 + 0.4 * n, 0, 1.4), 0, 0.97) * fade
        temp = (e ** 0.25) * g * 0.95 + 0.1 * n
        col = palette(temp)
        w = (1 - acc[m]) * alpha
        rgb[m] += w[:, None] * col * I[:, None]
        acc[m] += w
    return rgb, (1 - acc) * star

def to_cells(rgb):
    img = rgb.reshape(R, SS, C, SS, 3)
    lum_s = img @ np.array([0.3, 0.55, 0.15])
    lum = lum_s.mean(axis=(1, 3))
    colsum = (img * lum_s[..., None]).sum(axis=(1, 3)) / (lum_s.sum(axis=(1, 3))[..., None] + 1e-9)
    return lum, colsum

def frame(t, exposure=3.1):
    rgb, st = shade(t)
    lum, col = to_cells(rgb)
    lum = lum + 0.22 * gaussian_filter(lum, (2.0, 3.4)) + 0.10 * gaussian_filter(lum, (6, 10))  # bloom
    L = 1 - np.exp(-exposure * lum)               # tone map 0..1
    stc = st.reshape(R, SS, C, SS).max(axis=(1, 3))
    idx = np.clip((L ** 0.75) * (len(RAMP) - 1) * 1.02, 0, len(RAMP) - 1).astype(int)
    chars = RAMP_ARR[idx]
    starmask = (L < 0.06) & (stc > 0.05)
    chars = np.where(starmask, np.where(stc > 0.55, "*", np.where(stc > 0.2, "+", ".")), chars)
    # color brightness follows tone, hue follows physics
    cmax = col.max(-1, keepdims=True) + 1e-9
    c = col / cmax * (0.35 + 0.65 * np.sqrt(L))[..., None]
    c = np.where(starmask[..., None], (np.array([0.75, 0.8, 0.95]) * (0.5 + 0.5 * stc[..., None])), c)
    c = (np.clip(c, 0, 1) * 255).astype(np.uint8)
    return chars, c, L

if __name__ == "__main__":
    from PIL import Image, ImageDraw, ImageFont
    chars, c, L = frame(0.0)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10)
    cw, chh = 6, 10
    im = Image.new("RGB", (C * cw, R * chh), (13, 17, 23)); d = ImageDraw.Draw(im)
    for y in range(R):
        for x in range(C):
            if chars[y, x] != " ":
                d.text((x * cw, y * chh - 1), chars[y, x], font=font, fill=tuple(int(v) for v in c[y, x]))
    im.save("preview.png")
    print("saved", im.size)


# ---------- shooting comets (fully inside the loop, so it stays seamless) ----------
COMETS = [dict(t0=0.06, dur=0.30, x0=150, y0=-3, vx=-150, vy=26),
          dict(t0=0.52, dur=0.26, x0=245, y0=8, vx=-120, vy=30),
          dict(t0=0.78, dur=0.18, x0=60, y0=-2, vx=-70, vy=18)]
TAIL = "@#*+=~-:.'`"

def add_comets(chars, col, t):
    chars = chars.copy(); col = col.copy()
    for c in COMETS:
        s = (t - c["t0"]) / c["dur"]
        if not (0 <= s <= 1): continue
        hx, hy = c["x0"] + c["vx"] * s, c["y0"] + c["vy"] * s
        sp = np.hypot(c["vx"], c["vy"] / 0.6)
        ux, uy = -c["vx"] / sp, -c["vy"] / sp           # tail points backwards
        n = 52
        for k in range(n):
            x = int(round(hx + ux * k * 1.0)); y = int(round(hy + uy * k * 0.6))
            if 0 <= y < chars.shape[0] and 0 <= x < chars.shape[1]:
                f = k / n
                chars[y, x] = TAIL[min(len(TAIL) - 1, int(f ** 0.6 * len(TAIL)))] if k else "@"
                b = (1 - f) ** 1.4
                col[y, x] = (np.array([235, 245, 255]) * b + np.array([60, 110, 170]) * (1 - b) * (1 - f)).clip(0, 255)
        # a little glow beside the head
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = int(round(hx)) + dx, int(round(hy)) + dy
            if 0 <= y < chars.shape[0] and 0 <= x < chars.shape[1] and chars[y, x] == " ":
                chars[y, x] = "."; col[y, x] = (150, 190, 230)
    return chars, col
