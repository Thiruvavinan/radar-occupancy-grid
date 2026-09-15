# Radar Occupancy Grid

Detecting objects from automotive radar and telling **static from moving** — by
accumulating sparse returns into a persistent bird's-eye grid instead of
clustering each sweep on its own. Built on nuScenes.

![BEV output](outputs/scene-1077/frame_036.png)

*Amber = nuScenes annotations. Circles = static returns, triangles = moving.
Grey = grid cells. Teal = static objects, red = moving (arrow is 1 s of travel).*

## What it does

One radar keyframe gives ~190 points over a 120 m square, and any single return
may be noise — too sparse to cluster directly.

So every return goes into a 1 m grid cell, and each cell accumulates evidence
across frames: a hit count, running means of RCS / height / velocity, and a
probability that something is really there. Hits raise it, misses lower it.
Clutter fades and is deleted; real structure persists. Only confident cells get
clustered into objects.

```
run.py
 └── radar/ingest.py   load 5 radars, sensor → ego → global
     radar/grid.py     accumulate: hits, decay, aging, P(existence)
     radar/detect.py   filter → DBSCAN over cells → objects
     radar/viz.py      BEV figure
```

## How it works

One cell, watched over 12 frames. This is the whole idea:

```
frame  hits  log-odds  P(exists)
    0     1     0.847      0.700   HIT
    1     1     0.442      0.609   miss
    3     1    -0.369      0.409   miss    ← fading, nearly deleted
    4     2     0.478      0.617   HIT     ← passes the filter
    8     5     2.615      0.932   HIT     ← strongly believed
   11     5     1.398      0.802   miss    ← decays, survives on past evidence
```

Confidence is a **log-odds binary Bayes filter**, so evidence just adds:

```
hit    →  l ← min(l + 0.847, +4.0)      # log(0.70/0.30)
miss   →  l ← max(l − 0.405, −2.0)      # log(0.40/0.60)
p = 1 − 1/(1 + e^l)
```

A cell is deleted after 6 frames without a hit, or below `l = −1.5`.

**Two motions are compensated.** Ego motion: cells are keyed in the global map
frame, so a parked car holds the same cell however the vehicle drives. Object
motion: moving cells are advected to `center + v·dt`, without which a car at
10 m/s crosses five cells per frame and is never detected.

**Objects draw from two sources on purpose** — position, extent and confidence
from the accumulated grid; **velocity from the current frame only**, since a
running mean would still report a stopped car as moving.

## Results

Three scenes, 121 keyframes, ~0.17 s/frame.

| | scene-0061 | scene-0757 | scene-1077 |
|---|---|---|---|
| objects / frame | 36.9 | 30.0 | 26.0 |
| **mean speed, moving** | **3.78 m/s** | **3.68 m/s** | **7.63 m/s** |
| **mean speed, static** | **0.09 m/s** | **0.05 m/s** | **0.16 m/s** |
| annotations covered (≤4 m) | 66.1% | 61.8% | 56.1% |
| **static/dynamic correct** | **87.0%** | **91.2%** | **96.4%** |

![static vs moving](outputs/scene-1077/static_vs_moving.png)

Clean bimodal separation with a gap at the threshold. Accumulation also
localises better than the raw returns it is built from — median distance to the
nearest annotation drops from 1.30 m (raw points) to 1.12 m (grid cells).

Advection's benefit scales with speed, as expected:

| object speed | advected | not compensated | gain |
|---|---|---|---|
| 2–5 m/s | 1.97 m | 2.43 m | 19% |
| 5–10 m/s | 1.28 m | 4.13 m | **69%** |
| >10 m/s | 1.38 m | 6.50 m | **79%** |

## Limitations

- **Not a detection benchmark.** Nearest-centroid proximity with a 4 m gate —
  no IoU, no nuScenes eval protocol. Don't compare to published numbers.
- **Pedestrians are found but called static** (~1.6 returns each at walking
  pace); two-wheelers are missed. Physics of a 2 Hz low-return sensor, not a
  clustering bug — and why radar supports a lidar-primary stack rather than
  replacing it.
- **Object count isn't calibrated** — 26–37/frame vs 9–20 annotations. Much is
  real unannotated structure (walls, barriers), but how much isn't measured.
- **Tuned on the scenes it reports on.** No held-out set. No tracking.
- **Ego pose is taken straight from the keyframe** — nuScenes is synchronised,
  so no SLERP interpolation is needed. Asynchronous sensors would require it.

## Run it

nuScenes mini (~4 GB) isn't included — download from
[nuscenes.org](https://www.nuscenes.org/nuscenes#download) and unpack so
`v1.0-mini/` sits in the repo root.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install --no-deps nuscenes-devkit
```

`--no-deps` is deliberate: the devkit's pinned dependencies drag in
`pycocotools`, which needs a C toolchain on Windows and is never used here.

```bash
python run.py --list-scenes
python run.py --scenes scene-0061 scene-0757 scene-1077
python run.py --scenes scene-1077 --no-advection   # watch movers disappear
python run.py --scenes scene-1077 --anchor ego     # vehicle-centred grid
```

Start reading at [`run_scene`](run.py#L127) — the whole pipeline is ~20 lines —
then [`grid.update`](radar/grid.py#L184).

---

Data: nuScenes mini split. Caesar et al., *nuScenes: A multimodal dataset for
autonomous driving*, CVPR 2020.
