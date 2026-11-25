"""Ray-trace a Schwarzschild black hole (rs = 1) with a thin accretion disk.

Camera is static, so geometry is traced once; the animation only re-samples
the disk texture. Output: geometry.npz with per-ray disk hits + escaped-ray
directions for a lensed starfield.
"""
import numpy as np, sys, time

C, R = int(sys.argv[1]), int(sys.argv[2])     # character grid
SS = 3                                         # supersample per cell (SSxSS)
CELL_ASPECT = 0.6                              # char width / height
D, ELEV, ROLL = 24.0, np.radians(9.0), np.radians(-7.0)
TAN_H = np.tan(np.radians(34.0))
R_IN, R_OUT = 3.0, 15.0
MAXHIT = 3

W, H = C * SS, R * SS
cam = np.array([0.0, D * np.sin(ELEV), -D * np.cos(ELEV)])
fwd = -cam / np.linalg.norm(cam)
right = np.cross(np.array([0.0, 1.0, 0.0]), fwd); right /= np.linalg.norm(right)
up = np.cross(fwd, right)
cr, sr = np.cos(ROLL), np.sin(ROLL)
right, up = cr * right + sr * up, -sr * right + cr * up

xs = (np.arange(W) + 0.5) / W * 2 - 1
ys = (np.arange(H) + 0.5) / H * 2 - 1
vis_aspect = H / (W * CELL_ASPECT)
U, V = np.meshgrid(xs * TAN_H, -ys * TAN_H * vis_aspect + 0.02)
dirs = fwd + U[..., None] * right + V[..., None] * up
dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

N = W * H
pos = np.tile(cam, (N, 1)).astype(np.float64)
vel = dirs.reshape(N, 3).astype(np.float64)
h2 = np.sum(np.cross(pos, vel) ** 2, axis=1)

hit_r = np.zeros((MAXHIT, N)); hit_phi = np.zeros((MAXHIT, N)); hit_g = np.zeros((MAXHIT, N))
nhit = np.zeros(N, int)
alive = np.ones(N, bool); captured = np.zeros(N, bool)

def accel(p, h2):
    r2 = np.sum(p * p, axis=1)
    return -1.5 * h2[:, None] * p / (r2 ** 2.5)[:, None]

t0 = time.time()
idx = np.arange(N)
for it in range(4000):
    a = idx[alive]
    if a.size == 0: break
    p, v, hh = pos[a], vel[a], h2[a]
    r = np.linalg.norm(p, axis=1)
    dt = np.clip(0.06 * (r - 0.6), 0.01, 0.8)
    # velocity-Verlet
    acc = accel(p, hh)
    pn = p + v * dt[:, None] + 0.5 * acc * dt[:, None] ** 2
    vn = v + 0.5 * (acc + accel(pn, hh)) * dt[:, None]
    # disk crossing
    cross = (p[:, 1] * pn[:, 1]) < 0
    if cross.any():
        ci = np.nonzero(cross)[0]
        f = p[ci, 1] / (p[ci, 1] - pn[ci, 1])
        hp = p[ci] + (pn[ci] - p[ci]) * f[:, None]
        hr = np.hypot(hp[:, 0], hp[:, 2])
        ok = (hr > R_IN) & (hr < R_OUT)
        ci, hp, hr = ci[ok], hp[ok], hr[ok]
        gi = a[ci]
        room = nhit[gi] < MAXHIT
        ci, hp, hr, gi = ci[room], hp[room], hr[room], gi[room]
        if gi.size:
            phi = np.arctan2(hp[:, 2], hp[:, 0])
            beta = np.sqrt(0.5 / (hr - 1.0))
            gam = 1 / np.sqrt(1 - beta ** 2)
            tang = np.stack([-np.sin(phi), np.zeros_like(phi), np.cos(phi)], 1)
            vv = v[ci] / np.linalg.norm(v[ci], axis=1, keepdims=True)
            n = -vv                                   # photon heads to camera
            delta = 1 / (gam * (1 - beta * np.sum(tang * n, 1)))
            g = delta * np.sqrt(1 - 1 / hr)
            k = nhit[gi]
            hit_r[k, gi] = hr; hit_phi[k, gi] = phi; hit_g[k, gi] = g
            nhit[gi] += 1
    pos[a], vel[a] = pn, vn
    rn = np.linalg.norm(pn, axis=1)
    cap = rn < 1.0
    esc = (rn > 60) & (np.sum(pn * vn, 1) > 0)
    full = nhit[a] >= MAXHIT
    captured[a[cap]] = True
    alive[a[cap | esc | full]] = False
    if it % 200 == 0:
        print(it, alive.sum(), f"{time.time()-t0:.1f}s", flush=True)

escaped = ~captured
d = vel / np.linalg.norm(vel, axis=1, keepdims=True)
np.savez_compressed("geometry.npz", C=C, R=R, SS=SS, hit_r=hit_r, hit_phi=hit_phi,
                    hit_g=hit_g, nhit=nhit, escaped=escaped, edir=d, R_IN=R_IN, R_OUT=R_OUT)
print("done", f"{time.time()-t0:.1f}s", "hits:", np.bincount(nhit))
