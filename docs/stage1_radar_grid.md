# Stage 1 — design rationale, update rules and evidence

Depth for anyone who wants it; the README is the two-minute version.

---

## 1. Coordinate convention, verified

ISO 8855: **x forward, y left, z up**. (ISO 8855 is the vehicle-dynamics axis
standard; other numbers sometimes cited for this are wrong.)

nuScenes' ego frame is documented as using it, but "documented" is not
"verified", and a silent rotation error would corrupt every downstream stage
while still producing plausible-looking plots. Two independent checks,
implemented in `radar/transform.py:verify_convention` and runnable with
`--verify`:

| check | result |
|---|---|
| cos angle between ego +x and direction of travel | mean **+0.9996**, min +0.9985 (38 frames) |
| RADAR_FRONT mounted at | x = **+3.41 m** (forward → positive x) |
| RADAR_FRONT_LEFT mounted at | y = **+0.80 m** (left → positive y) |

**No rotation correction is applied**, because none is needed. This was checked
before any transform code was written, not after results looked wrong.

## 2. Multi-radar time synchronisation

The measurement that justifies the module, over 202 keyframes of 5 scenes:

| sweep spread across the five radars | |
|---|---|
| median | **48.8 ms** |
| p90 | 65.4 ms |
| max | **73.3 ms** |
| max, expressed as ego travel at 15 m/s | **1.10 m** |

A metre of unmodelled displacement at 1 m cell resolution puts a static object
in the wrong cell, which is exactly what destroys accumulation. So each radar is
carried to the common frame through its own pose:

```
p_global = T_ego→global(t_radar) · T_sensor→ego · p_sensor
p_ref    = T_global→ego(t_ref)   · p_global
```

`T_ego→global(t)` comes from `EgoPoseInterpolator`:

- **translation**: linear interpolation between the bracketing poses
- **rotation**: **SLERP** on the quaternion — componentwise lerp of two
  quaternions leaves the unit sphere and gives a non-uniform angular rate,
  whereas SLERP traces the shortest arc at constant angular velocity
- **exact timestamps are returned untouched**, since interpolating onto a
  recorded pose only adds floating-point noise
- **outside the recorded range, clamp rather than extrapolate** — inventing
  motion we have no evidence for is worse than a stale pose

Poses are gathered per scene through `sample_data`, never from the global
`ego_pose` table, so poses from a different log can never be interpolated
against each other.

**Honest caveat.** nuScenes stores an ego pose at every `sample_data`
timestamp (verified: 100/100 records checked), median spacing 5.9 ms. So for
radar sweeps the lookup resolves exactly and SLERP is a no-op in practice. It is
needed for an arbitrary reference time, and it is what a production stack with
genuinely asynchronous sensors requires. Velocity uses `vx_comp`/`vy_comp` — the
ego-motion-compensated fields — not the raw radial velocity in rows 6–7.

## 3. The occupancy update rule

A log-odds binary Bayes filter. Each cell holds `l = log(p/(1−p))`, updated once
per cycle:

```
hit          (≥1 return in the cell):      l ← min(l + 0.847, +4.0)
miss         (no return, cell within 50 m): l ← max(l − 0.405, −2.0)
out of range (cell beyond 50 m):            l unchanged
```

from `p_hit = 0.70` and `p_miss = 0.40`; prior `p = 0.5`, so a fresh cell starts
at `l = 0`. Readout is the standard logistic `p = 1 − 1/(1 + e^l)`, bounding `p`
to [0.119, 0.982].

Why each piece:

- **Log-odds rather than probability** — evidence combines by addition, and `p`
  can never reach exactly 0 or 1 and freeze.
- **`|l_hit| > |l_miss|`** — a radar miss is weak evidence (occlusion, unlucky
  aspect angle, a low-RCS moment); a hit is comparatively strong.
- **The +4.0 clamp is load-bearing** — without it a wall seen for fifty frames
  needs fifty frames of misses to forget, long after leaving the sensor's view.
  The clamp bounds how much history a cell can hoard, keeping it revisable.
- **Within-cycle aggregation** — a cell catching five returns in one sweep is
  **one** observation. Counting five would let dense clutter manufacture
  confidence, which is precisely what the grid exists to prevent.
- **The miss update is not ray-cast free space.** A full occupancy grid traces
  each beam and marks everything in front of a return empty. Radar's angular
  resolution makes that region unreliable, so free space is never marked free,
  only un-reinforced. This under-uses the sensor and is a deliberate
  simplification.

Aging: deleted after 6 cycles (3 s) without a hit, below `l = −1.5`, or once
outside 1.2 × the active radius (the 1.2 is hysteresis, so a boundary cell is
not deleted and recreated on alternating frames).

## 4. Object-motion compensation (advection)

A world-anchored grid accumulates static objects well and moving ones not at
all: at 2 Hz a vehicle at 10 m/s crosses ten 1 m cells between cycles, never
revisits one, and never reaches a second hit. So before each update every cell
with accumulated speed ≥ 0.5 m/s is carried to `center + v·dt`. Slower cells are
held still — advecting them would inject velocity noise into the static world,
which already works. Where two cells land on one square the better-supported one
survives.

## 5. Parameter selection

Chosen on **mini_train**, reported once on **mini_val**, so the headline numbers
are not fitted to the split they are quoted on. Class-agnostic recall/precision
at 4 m on mini_train:

| configuration | recall | precision | NDS |
|---|---|---|---|
| res 0.5 m, `min_samples` 2, `l ≥ 0.7` | 18.5% | 38.4% | 0.0001 |
| res 0.5 m, `min_samples` 1, `l ≥ 0.4` | 42.6% | 24.9% | 0.0096 |
| **res 1.0 m, `min_samples` 1, `l ≥ 0.4`** | **44.4%** | 23.7% | **0.0121** |

Two findings worth stating:

- **`min_samples = 1`** is unusual and is the single biggest lever. With 2,
  DBSCAN discards isolated confident cells as noise and recall at 4 m falls from
  44.4% to 18.5%. Radar is sparse enough that a cell which already cleared the
  log-odds and hit-count bars is real evidence; noise rejection belongs in that
  filter, not in DBSCAN.
- **A 0.5 m grid is slightly worse than 1.0 m.** Finer cells mostly split
  evidence that should have accumulated together.

## 5b. How much motion-awareness does the pipeline actually need?

Velocity is attached to every cell and every object, so `is_moving` is a derived
property (`speed > 0.5 m/s`), stored nowhere. The fair question is whether the
*pipeline* needs a static/dynamic distinction at all, or whether velocity alone
suffices. Measured on mini_val at a 4 m match:

| | precision | recall | motion call |
|---|---|---|---|
| **as shipped** | 56.6% | 35.3% | **94.7%** |
| without the per-cell hit rule (`min_hits_dynamic = 2`) | 58.9% | **31.5%** | 94.5% |
| without the DBSCAN motion split | 57.0% | 35.8% | 94.0% |
| without either | 58.6% | 31.6% | 94.5% |

Two of the three motion-aware mechanisms were **removed as a result**:

- **The DBSCAN motion split earned nothing.** It existed so a car passing parked
  cars would not merge with them; measured, it costs 0.5 points of recall to buy
  0.7 points on the motion call. Checked with road masking off as well, in case
  the mask had made it redundant — it had not; the split was simply never
  pulling its weight. Gone.
- **The advection gate earned nothing.** Advecting only cells above 0.5 m/s
  versus advecting every cell by its own velocity gives identical results
  (56.7% vs 56.6% precision). Every cell now moves by `v·dt` and a static cell
  stays put because its `v` is ~0. One rule instead of two.

What survives is **`min_hits_dynamic`**, and it is worth being precise about
what it is: not a classification, but a rule about *evidence*. A cell carrying a
coherent ground-referenced velocity is admitted on one observation rather than
two, because a moving target is structurally harder to observe twice in the same
place and a consistent Doppler reading is itself corroboration that clutter does
not produce. It buys 3.8 points of recall for 2.3 of precision.

Note also that the motion call sits at ~94.5% in *every* configuration above —
it barely depends on any of this machinery, because it comes from current-frame
Doppler regardless. That is the real answer to "why classify at all": mostly you
should not, and the pipeline now does so in exactly one place.

## 6. Why the official mAP is ~0, in detail

This was diagnosed rather than assumed, because "our mAP is low" is not an
explanation.

**Cause 1 — the class mapping.** With the intuitive velocity rule
(moving → `car`, static → `barrier`), mini_val contains **zero `barrier` GT
boxes**, so all 619 static detections were unmatchable by construction, and only
117 moving detections could score at all against 2056 GT cars — a ceiling of
5.7% recall. Switching to "everything is a `car`" is not metric-gaming: a parked
car *is* a car, and with no semantic information the maximum-likelihood class is
the most common one.

**Cause 2 — nuScenes' recall floor.** `calc_ap` clips recall below
`min_recall = 0.1` and precision below `min_precision = 0.1`. Measured directly
from the devkit's own `accumulate`, the earlier configuration reached precision
> 0 only up to recall **0.100** at the 4 m threshold and 0.080 at 2 m — landing
exactly on the cliff edge, so AP evaluated to exactly 0. Relaxing the filter
(section 5) lifted recall above the floor and AP became non-zero, if still
small.

So the ~0 mAP is two things stacked: 9 of 10 classes structurally absent, and a
car AP that is genuinely low for a classical detector. Learned detectors on this
benchmark reach mAP ≈ 0.5–0.6; a non-learned radar-only baseline reaching
`car AP = 0.012` is the honest number, not a bug.

**What the TP errors say.** mAVE, per-class velocity error on car, is
**0.184 m/s** — Doppler is the one quantity radar measures precisely, and it
shows. mASE 0.963 and mAOE 1.079 rad are near their worst possible values,
which is expected: size is a cluster extent with a hardcoded 1.5 m height, and
orientation is yaw-from-velocity with identity when static.

## 7. Detection field provenance

| field | source | honest status |
|---|---|---|
| translation x, y | confidence-weighted centroid of cluster cells | measured |
| translation z | **0.0** | placeholder — radar does not resolve height |
| size w, l | cluster extent, floored at 1.5 m | weak estimate |
| size h | **1.5 m** | hardcoded constant |
| rotation | yaw from velocity; identity when static | derived, not measured |
| velocity | mean compensated velocity of the **current** cycle's returns | measured, and good |
| score | mean occupancy probability of the cluster's cells | derived |
| class | single-label rule | not classification |

Velocity deliberately comes from the current cycle, not the grid's running mean:
a running mean lags and would report a stopped vehicle as still moving, and
braking is exactly when a downstream consumer needs the truth.

## 8. Road masking trade-off

`drivable_area`, `road_segment` and `lane` polygons are loaded once per map
location and indexed with an STRtree; cells whose global centre falls outside
are dropped before clustering.

**How aggressive it is:** only **17.1% of radar returns** fall on the drivable
surface, so roughly two thirds of detections are removed. Measured on mini_val:

| | mask off | mask on |
|---|---|---|
| detections | 2601 | 864 |
| precision @ 4 m | 24.6% | **56.6%** |
| recall @ 4 m | **46.1%** | 35.3% |
| precision @ 2 m | 17.1% | **44.2%** |
| recall @ 2 m | **32.2%** | 27.5% |
| car AP | 0.0119 | **0.0326** |
| mAP | 0.0012 | **0.0033** |
| NDS | 0.0148 | 0.0158 |

Precision more than doubles, car AP nearly triples, and it costs **10.8 points
of recall at 4 m**. That recall is not noise being removed — it is real objects
off the drivable surface: pedestrians on pavements, cyclists on bike paths, cars
parked half on a kerb. The filter is defensible for a ground-vehicle-focused
baseline and is on by default because the official metric prefers it, but the
lost recall is a real cost, not a rounding error.

Correctness was sanity-checked before trusting the numbers: the ego vehicle's
own position lands on the drivable area in 20/20 frames, and the same positions
shifted 300 m land on it 0/20 times. Masking costs ~7 ms per frame.

Requires the map expansion pack at `maps/expansion/*.json`; the module detects
its absence and disables itself rather than failing.
