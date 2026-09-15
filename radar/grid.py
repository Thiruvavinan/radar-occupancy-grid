"""Step 4: accumulate sparse radar returns into a persistent occupancy grid.

This is the heart of the pipeline. A single radar sweep gives ~200 noisy points
over a 120 m square -- far too sparse to cluster directly. Instead every return
is dropped into a 1 m cell, and each cell remembers how often it has been hit,
how confident we are that something is really there, and the running mean of
what was measured. Clutter fails to repeat and decays away; real structure
repeats and climbs.

Two things have to be compensated for, and they are easy to confuse:

  **ego motion**  -- handled by keying cells in the *global map frame*, so a
                     parked car keeps the same cell index however the ego
                     drives. (`anchor="ego"` keys cells to the vehicle instead
                     and drags the whole grid along each frame; see below.)

  **object motion** -- handled by `_advect`, which carries a moving cell to
                     where its own velocity says it will be next frame.
                     Without this, a car at 10 m/s crosses five cells between
                     keyframes, never revisits one, and is never detected.

Storage is a sparse dict, not a dense array: only a few thousand cells are ever
live, while a dense +/-60 m array at 1 m would be 14,400 mostly-empty cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from radar.ingest import transform_points


def prob_to_log_odds(p: float) -> float:
    return math.log(p / (1.0 - p))


def log_odds_to_prob(l):
    return 1.0 - 1.0 / (1.0 + np.exp(l))


#: Compensated ground speed above which a cell or object counts as moving.
MOVING_SPEED = 0.5   # m/s


@dataclass
class GridConfig:
    """Every tunable of the accumulation stage, in one place."""

    resolution: float = 1.0          # metres per cell
    active_radius: float = 60.0      # keep cells within this range of ego

    # "world": cell index is a map coordinate, so a stationary object keeps its
    #          index and accumulation never resamples.
    # "ego":   cell index is relative to the vehicle, giving bounded
    #          vehicle-centred storage, but every cell must be re-keyed (and so
    #          re-rounded) each frame.
    # Measured over six scenes these perform the same end to end; the ego
    # anchor loses ~15% of accumulation depth but detections are within a
    # point. "world" is the default for being exact and simpler.
    anchor: str = "world"

    # --- existence probability: a log-odds binary Bayes filter -----------
    p_hit: float = 0.70              # -> l_hit  = +0.847
    p_miss: float = 0.40             # -> l_miss = -0.405
    p_prior: float = 0.50            # a fresh cell starts at l = 0
    log_odds_max: float = 4.0        # clamp -> p <= 0.982
    log_odds_min: float = -2.0       # clamp -> p >= 0.119

    # --- aging and deletion ---------------------------------------------
    max_frames_since_hit: int = 6    # 3 s at 2 Hz
    delete_below_log_odds: float = -1.5

    # --- object-motion compensation --------------------------------------
    advect_dynamic_cells: bool = True

    # --- filtering before clustering -------------------------------------
    min_hits: int = 2                # a static cell must be seen twice
    # Moving cells are admitted on ONE observation. A mover is structurally
    # harder to see twice in one place, and a coherent ego-compensated velocity
    # is itself corroboration -- clutter does not produce a consistent speed
    # over ground. Raising this to 2 drops moving-car coverage from 50% to 17%,
    # which makes it the most important number in this file.
    min_hits_dynamic: int = 1
    min_existence_prob: float = 0.60

    @property
    def l_hit(self):
        return prob_to_log_odds(self.p_hit)

    @property
    def l_miss(self):
        return prob_to_log_odds(self.p_miss)

    @property
    def l_prior(self):
        return prob_to_log_odds(self.p_prior)


@dataclass
class Cell:
    """One accumulated cell. Means are running means over hits only."""

    i: int
    j: int
    log_odds: float
    hit_count: int = 0
    last_hit_frame: int = -1
    mean_rcs: float = 0.0
    mean_z: float = 0.0
    mean_vx: float = 0.0             # velocity, in the anchor frame
    mean_vy: float = 0.0

    @property
    def p_exist(self) -> float:
        return float(log_odds_to_prob(self.log_odds))

    @property
    def speed(self) -> float:
        return math.hypot(self.mean_vx, self.mean_vy)


@dataclass
class CellSnapshot:
    """Columnar view of a set of cells, for the vectorised stages downstream."""

    centers_global: np.ndarray       # (M, 2)
    centers_ego: np.ndarray          # (M, 2)
    p_exist: np.ndarray              # (M,)
    hit_count: np.ndarray            # (M,)
    mean_rcs: np.ndarray             # (M,)
    mean_z: np.ndarray               # (M,)
    mean_velocity: np.ndarray        # (M, 2) always global frame

    def __len__(self) -> int:
        return int(self.centers_global.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


def _empty_snapshot() -> CellSnapshot:
    z2, z1 = np.zeros((0, 2)), np.zeros(0)
    return CellSnapshot(z2, z2, z1, np.zeros(0, dtype=int), z1, z1, z2)


class RadarOccupancyGrid:
    """Accumulates radar returns into persistent, confidence-weighted cells."""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self.cells: dict = {}
        self.frame_index = -1
        self.last_timestamp = None
        self.last_ego_to_global = None

    # -- indexing and the anchor frame ------------------------------------

    def _to_index(self, xy):
        return np.floor(np.asarray(xy) / self.config.resolution).astype(np.int64)

    def _to_center(self, indices):
        return (np.asarray(indices, dtype=float) + 0.5) * self.config.resolution

    @property
    def _ego_anchored(self) -> bool:
        return self.config.anchor == "ego"

    def _anchor_points(self, frame):
        return frame.points_ego if self._ego_anchored else frame.points_global

    def _anchor_velocity(self, frame):
        return frame.velocity_ego if self._ego_anchored else frame.velocity_global

    def _anchor_ego_xy(self, frame):
        """Where the ego is in the anchor frame -- the origin, if ego-anchored."""
        return np.zeros(2) if self._ego_anchored else frame.ego_translation[:2]

    # -- the per-frame update ---------------------------------------------

    def update(self, frame) -> None:
        """Fold one RadarFrame into the grid.

        0. re-anchor (ego-anchored only) so cells keep up with the vehicle
        1. advect moving cells along their own velocity
        2. hit  update: log-odds up, running means refreshed
        3. miss update: log-odds down for in-range cells that got nothing
        4. age out and delete
        """
        self.frame_index = frame.frame_index
        self._reanchor(frame)
        self._advect(frame)
        hit = self._apply_hits(frame)
        self._apply_misses(frame, hit)
        self._prune(frame)
        self.last_timestamp = frame.timestamp
        self.last_ego_to_global = frame.ego_to_global.copy()

    def _reanchor(self, frame) -> None:
        """Ego-anchored only: drag every cell along to follow the vehicle.

        This is what the ego anchor costs. Re-keying rounds to the nearest cell
        every frame, so quantisation error compounds, and a yaw change swings
        distant cells by more than a cell width. The world anchor never
        resamples at all.
        """
        if not self._ego_anchored or self.last_ego_to_global is None or not self.cells:
            return

        relative = frame.global_to_ego @ self.last_ego_to_global   # old ego -> new ego
        keys = np.array(list(self.cells.keys()), dtype=float)
        centers = self._to_center(keys)
        padded = np.hstack([centers, np.zeros((centers.shape[0], 1))])
        new_keys = self._to_index(transform_points(padded, relative)[:, :2])
        rotation = relative[:2, :2]     # velocities live in the anchor frame too

        rebuilt: dict = {}
        for cell, key in zip(self.cells.values(), map(tuple, new_keys.tolist())):
            vx, vy = rotation @ np.array([cell.mean_vx, cell.mean_vy])
            cell.mean_vx, cell.mean_vy = float(vx), float(vy)
            cell.i, cell.j = key
            rebuilt[key] = _keep_better(rebuilt.get(key), cell, key)
        self.cells = rebuilt

    def _advect(self, frame) -> None:
        """Carry moving cells to where their own velocity says they now are.

        A cell already knows its velocity from the returns that formed it, so
        it can predict its own next position: center + v*dt. Cells slower than
        MOVING_SPEED are held still -- advecting them would only inject
        velocity noise into the static world, which already works well.
        """
        if not self.config.advect_dynamic_cells or self.last_timestamp is None:
            return
        dt = frame.timestamp - self.last_timestamp
        if dt <= 0 or dt > 2.0:                 # guard against scene seams
            return

        # Static cells claim their squares first, so an arriving moving cell
        # merges rather than silently evicting one.
        moved = {k: c for k, c in self.cells.items() if c.speed < MOVING_SPEED}
        dynamic = [(k, c) for k, c in self.cells.items() if c.speed >= MOVING_SPEED]

        for key, cell in dynamic:
            predicted = self._to_center(key) + np.array([cell.mean_vx, cell.mean_vy]) * dt
            new_key = tuple(self._to_index(predicted).tolist())
            cell.i, cell.j = new_key
            moved[new_key] = _keep_better(moved.get(new_key), cell, new_key)
        self.cells = moved

    def _apply_hits(self, frame) -> set:
        """Log-odds increment and running-mean refresh for every observed cell."""
        cfg = self.config
        if frame.num_points == 0:
            return set()

        points = self._anchor_points(frame)
        velocities = self._anchor_velocity(frame)

        # Group this frame's returns by cell first. A cell catching five returns
        # in one sweep is still ONE observation of that cell -- counting it five
        # times would let dense clutter manufacture confidence.
        buckets: dict = {}
        for idx, key in enumerate(map(tuple, self._to_index(points[:, :2]).tolist())):
            buckets.setdefault(key, []).append(idx)

        for key, members in buckets.items():
            cell = self.cells.get(key)
            if cell is None:
                cell = self.cells[key] = Cell(key[0], key[1], cfg.l_prior)

            cell.log_odds = min(cell.log_odds + cfg.l_hit, cfg.log_odds_max)
            cell.hit_count += 1
            cell.last_hit_frame = frame.frame_index

            # This frame's mean, folded into the running mean: m += (x - m)/n
            velocity = velocities[members].mean(axis=0)
            n = cell.hit_count
            cell.mean_rcs += (float(np.mean(frame.rcs[members])) - cell.mean_rcs) / n
            cell.mean_z += (float(np.mean(points[members, 2])) - cell.mean_z) / n
            cell.mean_vx += (float(velocity[0]) - cell.mean_vx) / n
            cell.mean_vy += (float(velocity[1]) - cell.mean_vy) / n

        return set(buckets)

    def _apply_misses(self, frame, hit_cells: set) -> None:
        """Decay cells we could plausibly have seen this frame but did not.

        Note what this is not: a ray-cast free-space model. A full occupancy
        grid traces each beam and marks everything in front of a return as
        empty. Radar's angular resolution makes that unreliable, so instead any
        live in-range cell that got no return takes one miss step.
        """
        cfg = self.config
        ego_xy = self._anchor_ego_xy(frame)
        radius_sq = cfg.active_radius ** 2

        for key, cell in self.cells.items():
            if key in hit_cells:
                continue
            if float(np.sum((self._to_center(key) - ego_xy) ** 2)) > radius_sq:
                continue                        # out of range: not a real miss
            cell.log_odds = max(cell.log_odds + cfg.l_miss, cfg.log_odds_min)

    def _prune(self, frame) -> None:
        """Delete cells that aged out, lost confidence, or left the window."""
        cfg = self.config
        ego_xy = self._anchor_ego_xy(frame)
        radius_sq = (cfg.active_radius * 1.2) ** 2      # hysteresis at the edge

        for key in [k for k, c in self.cells.items()
                    if float(np.sum((self._to_center(k) - ego_xy) ** 2)) > radius_sq
                    or frame.frame_index - c.last_hit_frame > cfg.max_frames_since_hit
                    or c.log_odds <= cfg.delete_below_log_odds]:
            del self.cells[key]

    # -- readout ----------------------------------------------------------

    def _required_hits(self, cell: Cell) -> int:
        """Moving cells get the lower bar; see min_hits_dynamic."""
        cfg = self.config
        return (min(cfg.min_hits, cfg.min_hits_dynamic)
                if cell.speed >= MOVING_SPEED else cfg.min_hits)

    def snapshot(self, frame, filtered=False) -> CellSnapshot:
        """Columnar view of the live cells.

        ``filtered=False`` returns every live cell (what the heatmap draws).
        ``filtered=True`` applies the hit-count and existence-probability bars,
        i.e. step 5 -- only confident, persistent cells become object
        candidates.
        """
        cells = [c for c in self.cells.values()
                 if not filtered
                 or (c.hit_count >= self._required_hits(c)
                     and c.p_exist >= self.config.min_existence_prob)]
        if not cells:
            return _empty_snapshot()

        centers = self._to_center([[c.i, c.j] for c in cells])
        padded = np.hstack([centers, np.zeros((len(cells), 1))])
        velocities = np.array([[c.mean_vx, c.mean_vy] for c in cells])

        if self._ego_anchored:
            centers_ego = centers
            centers_global = transform_points(padded, frame.ego_to_global)[:, :2]
            # Cells store velocity in the anchor frame; downstream code always
            # wants global, so convert on the way out.
            velocities = (frame.ego_to_global[:2, :2] @ velocities.T).T
        else:
            centers_global = centers
            centers_ego = transform_points(padded, frame.global_to_ego)[:, :2]

        return CellSnapshot(
            centers_global=centers_global,
            centers_ego=centers_ego,
            p_exist=np.array([c.p_exist for c in cells]),
            hit_count=np.array([c.hit_count for c in cells], dtype=int),
            mean_rcs=np.array([c.mean_rcs for c in cells]),
            mean_z=np.array([c.mean_z for c in cells]),
            mean_velocity=velocities,
        )

    def to_ego_tensor(self, frame, half_size=60.0):
        """Rasterise into a fixed-size ego-centred (5, H, W) float32 tensor.

        Channels: [p_exist, mean_rcs, vx, vy, hit_count], vehicle at the centre.

        Worth knowing: a fixed-shape vehicle-centred tensor does NOT require
        anchor="ego". Rasterising on demand gives the bounded, constant-shape
        output a BEV network wants while accumulation underneath stays exact.
        """
        resolution = self.config.resolution
        size = int(round(2 * half_size / resolution))
        tensor = np.zeros((5, size, size), dtype=np.float32)

        snapshot = self.snapshot(frame)
        if snapshot.is_empty:
            return tensor

        rows = ((snapshot.centers_ego[:, 0] + half_size) / resolution).astype(int)
        cols = ((snapshot.centers_ego[:, 1] + half_size) / resolution).astype(int)
        inside = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
        rows, cols = rows[inside], cols[inside]

        velocity = (frame.global_to_ego[:2, :2] @ snapshot.mean_velocity[inside].T).T
        tensor[0, rows, cols] = snapshot.p_exist[inside]
        tensor[1, rows, cols] = snapshot.mean_rcs[inside]
        tensor[2, rows, cols] = velocity[:, 0]
        tensor[3, rows, cols] = velocity[:, 1]
        tensor[4, rows, cols] = snapshot.hit_count[inside]
        return tensor

    @property
    def num_active_cells(self) -> int:
        return len(self.cells)


def _keep_better(existing, cell, key):
    """Resolve two cells landing on one square, keeping the better supported."""
    if existing is None:
        return cell
    winner = cell if cell.hit_count > existing.hit_count else existing
    winner.log_odds = max(cell.log_odds, existing.log_odds)
    winner.i, winner.j = key
    return winner
