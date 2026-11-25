# Editing the space profile

**Change the words** — edit `profile.json`:
- `log`: the lines typed in the mission log (label, value)
- `stack`: the six names orbiting the star (in pairs: inner, middle, outer orbit)

Commit the change and the `space-profile` Action re-renders everything within a couple of minutes.
It also runs every night, so the constellation and telemetry stay current.

**Run it locally** (optional):
```
pip install numpy scipy pillow
cd generator
python fetch_stats.py --sample     # made-up data; on GitHub the Action uses your real data
python contrib.py && python telemetry.py && python mission.py profile.json && python orbit.py profile.json
```
SVGs land in `generator/out/`.

**The black hole** is rendered separately (it is static, so the Action doesn't touch it):
```
cd generator/blackhole
python trace.py 240 72      # ray-trace once
python export.py 48 10      # 48 frames at 10 fps -> out/blackhole.svg / .gif / .mp4
```
Knobs: `trace.py` camera tilt, zoom, disk size · `render.py` OMEGA0 (speed), exposure, palette, COMETS.
