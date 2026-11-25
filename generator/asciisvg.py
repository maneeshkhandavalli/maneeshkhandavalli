"""Tiny engine: shade a character grid -> dense ASCII frames -> lightweight animated SVG.

Every section is a "shader": f(t, X, Y) -> rgb (float, linear-ish 0..n) evaluated on a
supersampled grid. Frames are converted to ASCII, then the SVG stores one static base
layer (the most common value per cell) plus, per frame, only the cells that differ.
"""
import numpy as np
from scipy.cluster.vq import kmeans2
import os
os.makedirs("out", exist_ok=True)

RAMP = np.array(list(" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"))
CW, CH = 6, 10                 # cell size in SVG units (matches the hero)
BG = (13, 17, 23)
ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
TRANSPARENT = "\0"


def grid(C, R, ss=3, aspect=0.6):
    """Sub-sample coordinates in *cell units*: X across, Y down (Y scaled so 1 unit = 1 col width)."""
    xs = (np.arange(C * ss) + 0.5) / ss
    ys = (np.arange(R * ss) + 0.5) / ss / aspect
    return np.meshgrid(xs, ys)


def to_cells(rgb, R, C, ss=3):
    img = rgb.reshape(R, ss, C, ss, 3)
    lum_s = img @ np.array([0.3, 0.55, 0.15])
    lum = lum_s.mean(axis=(1, 3))
    col = (img * lum_s[..., None]).sum(axis=(1, 3)) / (lum_s.sum(axis=(1, 3))[..., None] + 1e-9)
    return lum, col


def asciify(lum, col, exposure=2.6, gamma=0.75, floor=0.035, min_bright=0.35, ramp=None):
    rp = RAMP if ramp is None else np.array(list(ramp))
    L = 1 - np.exp(-exposure * np.clip(lum, 0, None))
    idx = np.clip((L ** gamma) * (len(rp) - 1) * 1.02, 0, len(rp) - 1).astype(int)
    idx[L < floor] = 0
    chars = rp[idx]
    c = col / (col.max(-1, keepdims=True) + 1e-9) * (min_bright + (1 - min_bright) * np.sqrt(L))[..., None]
    return chars, (np.clip(c, 0, 1) * 255).astype(np.uint8)


def put_text(chars, cols, row, col0, text, rgb):
    for i, ch in enumerate(text):
        if 0 <= col0 + i < chars.shape[1]:
            chars[row, col0 + i] = ch
            cols[row, col0 + i] = rgb


def _quantize(frames, ncol):
    allc = np.concatenate([c[ch != " "] for ch, c in frames[:: max(1, len(frames) // 8)]]).astype(float)
    rng = np.random.default_rng(0)
    if len(allc) > 40000: allc = allc[rng.choice(len(allc), 40000, replace=False)]
    cent, _ = kmeans2(allc, min(ncol, len(np.unique(allc, axis=0))), seed=1, minit="++")
    cent = np.clip(cent, 0, 255).round().astype(int)
    out = []
    for ch, c in frames:
        d = ((c.reshape(-1, 1, 3).astype(float) - cent[None]) ** 2).sum(-1)
        out.append((ch, d.argmin(1).reshape(ch.shape)))
    return cent, out


def _row_text(y, x0, chars, ks, full_len=None):
    parts, cur, buf = [], None, ""
    for ch, k in zip(chars, ks):
        kk = None if ch == " " else k
        if ch == " " or kk == cur:
            buf += ESC.get(ch, ch)
        else:
            if buf: parts.append(f'<tspan class="q{cur}">{buf}</tspan>' if cur is not None else buf)
            cur, buf = kk, ESC.get(ch, ch)
    if buf.strip(): parts.append(f'<tspan class="q{cur}">{buf}</tspan>' if cur is not None else buf)
    if not parts: return ""
    n = full_len or len(chars)
    xa = f' x="{x0 * CW}"' if x0 else ""
    return f'<text{xa} y="{y * CH + 8}" textLength="{n * CW}">{"".join(parts)}</text>'


def build(frames, fps, ncol=18, extra_svg="", extra_css="", rx=10, merge_gap=3):
    """frames: list of (chars[R,C] str, rgb[R,C,3] uint8). Returns SVG text."""
    R, C = frames[0][0].shape
    cent, q = _quantize(frames, ncol)
    N = len(q)
    code = np.stack([np.char.add(ch, k.astype(str)) for ch, k in q])     # N,R,C
    # base = most common value per cell
    base_ch = q[0][0].copy(); base_k = q[0][1].copy()
    if N > 1:
        flat = code.reshape(N, -1)
        for i in range(flat.shape[1]):
            vals, cnt = np.unique(flat[:, i], return_counts=True)
            if len(vals) > 1:
                v = vals[cnt.argmax()]
                f = int(np.nonzero(flat[:, i] == v)[0][0])
                r, c = divmod(i, C)
                base_ch[r, c] = q[f][0][r, c]; base_k[r, c] = q[f][1][r, c]
    base = "".join(_row_text(y, 0, base_ch[y], base_k[y]) for y in range(R))
    layers = []
    for fi, (ch, k) in enumerate(q):
        diff = (ch != base_ch) | ((k != base_k) & (ch != " "))
        out = []
        for y in np.nonzero(diff.any(1))[0]:
            xs = np.nonzero(diff[y])[0]
            runs, s, e = [], xs[0], xs[0]
            for x in xs[1:]:
                if x - e <= merge_gap: e = x
                else: runs.append((s, e)); s = e = x
            runs.append((s, e))
            for s, e in runs:
                out.append(f'<rect x="{s*CW}" y="{y*CH}" width="{(e-s+1)*CW}" height="{CH}"/>')
                out.append(_row_text(y, s, ch[y, s:e + 1], k[y, s:e + 1]))
        layers.append(f'<g class="f" style="animation-delay:{fi / fps:.3f}s">{"".join(out)}</g>')
    dur = N / fps
    css_cols = "".join(f".q{i}{{fill:rgb({c[0]},{c[1]},{c[2]})}}" for i, c in enumerate(cent))
    anim = (f".f{{visibility:hidden;animation:k {dur:.3f}s step-end infinite}}"
            f"@keyframes k{{0%{{visibility:visible}}{100 / N:.4f}%{{visibility:hidden}}}}") if N > 1 else ""
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C*CW} {R*CH}" width="{C*CW}" height="{R*CH}" xml:space="preserve">
<style>
text{{font:9.6px ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace;white-space:pre}}
rect{{fill:rgb{BG}}}
{anim}
{css_cols}
{extra_css}
</style>
<rect width="100%" height="100%" rx="{rx}"/>
<g>{base}</g>
{"".join(layers) if N > 1 else ""}
{extra_svg}
</svg>'''
    return svg.replace('textLength="', 'lengthAdjust="spacingAndGlyphs" textLength="')


# ---------- shared shading helpers ----------
def periodic_noise(shape, sigma, seed, wrap_axis=1):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    modes = ["nearest", "nearest"]; modes[wrap_axis] = "wrap"
    n = gaussian_filter(rng.standard_normal(shape), sigma, mode=modes)
    return n / n.std()


def sample2d(tex, u, v):
    """Bilinear sample; u wraps (0..1), v clamps (0..1)."""
    H, W = tex.shape
    x = (u % 1.0) * W; y = np.clip(v, 0, 1) * (H - 1)
    x0 = np.floor(x).astype(int) % W; x1 = (x0 + 1) % W; fx = x - np.floor(x)
    y0 = np.floor(y).astype(int); y1 = np.minimum(y0 + 1, H - 1); fy = y - y0
    return (tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy) + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy


def ramp_color(t, stops, cols):
    t = np.asarray(t)
    out = np.empty(t.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(t, stops, np.asarray(cols, float)[:, c])
    return out / 255


def starfield(X, Y, seed, density=0.0016, t=None, twinkle=True):
    """Sparse point stars on the supersampled grid; twinkle loops with t in [0,1)."""
    gx = np.floor(X * 2).astype(np.int64); gy = np.floor(Y * 2).astype(np.int64)
    h = ((gx * 73856093) ^ (gy * 19349663) ^ (seed * 83492791)) % 1000003 / 1000003.0
    on = h < density
    b = np.where(on, 0.25 + 3.5 * (h / density) ** 3, 0.0)
    if twinkle and t is not None:
        ph = (h * 9871.0) % 1.0
        k = 1 + np.floor((h * 7919.0) % 3)                      # integer cycles per loop
        b = b * (0.55 + 0.45 * np.sin(2 * np.pi * (k * t + ph)))
    return b


def build_layers(base, layers, ncol=18, extra_svg="", extra_css="", rx=10):
    """base: (chars, rgb) static. layers: list of dict(frames=[(chars, rgb)], fps=, extra=[svg per frame] or None).
    Layer chars use TRANSPARENT ('\\0') for 'show what is below'. Later layers draw on top."""
    allf = [base] + [f for L in layers for f in L["frames"]]
    cent, q = _quantize([(np.where(c == TRANSPARENT, " ", c), col) for c, col in allf], ncol)
    R, C = base[0].shape
    bch, bk = q[0]
    out_base = "".join(_row_text(y, 0, bch[y], bk[y]) for y in range(R))
    css, groups, i = [], [], 1
    for li, L in enumerate(layers):
        N, fps = len(L["frames"]), L["fps"]
        css.append(f".L{li}{{visibility:hidden;animation:k{li} {N / fps:.3f}s step-end infinite}}"
                   f"@keyframes k{li}{{0%{{visibility:visible}}{100 / N:.4f}%{{visibility:hidden}}}}")
        for fi, (raw, _) in enumerate(L["frames"]):
            ch, k = q[i]; i += 1
            content = raw != TRANSPARENT
            ch = np.where(content, ch, bch); k = np.where(content, k, bk)
            out = []
            for y in np.nonzero(content.any(1))[0]:
                xs = np.nonzero(content[y])[0]
                runs, s, e = [], xs[0], xs[0]
                for x in xs[1:]:
                    if x - e <= 3: e = x
                    else: runs.append((s, e)); s = e = x
                runs.append((s, e))
                for s, e in runs:
                    out.append(f'<rect x="{s*CW}" y="{y*CH}" width="{(e-s+1)*CW}" height="{CH}"/>')
                    out.append(_row_text(y, s, ch[y, s:e + 1], k[y, s:e + 1]))
            if L.get("extra"): out.append(L["extra"][fi])
            groups.append(f'<g class="L{li}" style="animation-delay:{fi / fps:.3f}s">{"".join(out)}</g>')
    css_cols = "".join(f".q{j}{{fill:rgb({c[0]},{c[1]},{c[2]})}}" for j, c in enumerate(cent))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {C*CW} {R*CH}" width="{C*CW}" height="{R*CH}" xml:space="preserve">
<style>
text{{font:9.6px ui-monospace,SFMono-Regular,Menlo,Consolas,"DejaVu Sans Mono",monospace;white-space:pre}}
rect{{fill:rgb{BG}}}
{"".join(css)}
{css_cols}
{extra_css}
</style>
<rect width="100%" height="100%" rx="{rx}"/>
<g>{out_base}</g>
{"".join(groups)}
{extra_svg}
</svg>'''
    return svg.replace('textLength="', 'lengthAdjust="spacingAndGlyphs" textLength="')


def nebula(X, Y, seed, scale=30.0):
    """Soft static fbm nebula (rgb), cheap: built from blurred noise on the sample grid."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    H, W = X.shape
    acc = np.zeros((H, W))
    for s, w in ((scale, 1.0), (scale / 2.5, 0.5), (scale / 6, 0.25)):
        n = gaussian_filter(rng.standard_normal((H, W)), (s * 0.6, s)); acc += w * n / n.std()
    acc /= 1.75
    hue = gaussian_filter(rng.standard_normal((H, W)), (scale * 0.9, scale * 1.5)); hue /= hue.std()
    dens = np.clip(acc - 0.6, 0, None) ** 1.8
    c1 = np.array([0.35, 0.25, 0.75]); c2 = np.array([0.15, 0.55, 0.6]); c3 = np.array([0.8, 0.45, 0.25])
    w2 = np.clip(0.5 + 0.5 * hue, 0, 1)[..., None]
    col = c1 * (1 - w2) + c2 * w2
    col = col * 0.8 + c3 * np.clip(acc - 1.3, 0, 1)[..., None]
    return col * dens[..., None]
