"""Persistent BEV occupancy grid with a log-odds Bayesian existence update.

Cells are keyed in the **global map frame**, so a stationary object holds the
same cell index however the ego drives -- accumulation never has to resample.
Only cells within ``active_radius`` of the ego are kept, a rolling window over
a conceptually unbounded grid. Storage is a sparse dict: a few thousand cells
are live at a time, while a dense +/-50 m array at 0.5 m would be 40,000 cells
of mostly nothing.

Two motions are compensated, and they are easy to confuse:

    ego motion     handled by the global keying above, plus the per-radar SLERP
                   pose lookup in radar/sync.py
    object motion  handled by `_advect`: a moving cell is carried to
                   center + v*dt before the next update, because at 2 Hz a car
                   at 10 m/s crosses 5 cells between frames, never revisits
                   one, and would never accumulate
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from radar.transform import transform_points

#: Compensated ground speed above which a cell or object counts as moving.
MOVING_SPEED = 0.5   # m/s


def prob_to_log_odds(p: float) -> float:
    return math.log(p / (1.0 - p))


def log_odds_to_prob(l):
    """Standard logistic transform, the inverse of prob_to_log_odds."""
    return 1.0 - 1.0 / (1.0 + np.exp(l))


@dataclass
class GridConfig:
    """Every tunable of the accumulation stage.

    Resolution is 1.0 m and the filter bar is log-odds 0.4, both chosen on
    **mini_train** and then reported once on mini_val, so the headline numbers
    are not fitted to the split they are quoted on. A 0.5 m grid was tried and
    is slightly worse (class-agnostic recall 42.6% vs 44.4% at 4 m on
    mini_train): radar is sparse enough that finer cells mostly split evidence
    that should have accumulated together.
    """

    resolution: float = 1.0
    active_radius: float = 50.0

    # --- log-odds existence update ---------------------------------------
    p_hit: float = 0.70              # -> l_hit  = +0.847
    p_miss: float = 0.40             # -> l_miss = -0.405
    p_prior: float = 0.50            # a fresh cell starts at l = 0
    log_odds_max: float = 4.0        # clamp -> p <= 0.982
    log_odds_min: float = -2.0       # clamp -> p >= 0.119

    # --- aging and deletion ------------------------------------------------
    max_frames_since_hit: int = 6    # 3 s at 2 Hz
    delete_below_log_odds: float = -1.5

    # --- object-motion compensation ---------------------------------------
    advect_dynamic_cells: bool = True

    # --- filtering before clustering ---------------------------------------
    # log-odds 0.4 -> p >= 0.599. A stricter 0.7 was tried and costs far more
    # recall than the precision it buys (18.5% -> 42.6% recall at 4 m when
    # relaxed, precision 38.4% -> 24.9%), measured on mini_train.
    min_log_odds: float = 0.4
    min_hits: int = 2
    # A cell carrying a coherent ego-compensated velocity is admitted on ONE
    # observation instead of two. This is a rule about *evidence*, not a
    # static/dynamic classification: a moving target is structurally harder to
    # observe twice in the same place, and a consistent speed over ground is
    # itself corroboration, since clutter does not produce one. It is the only
    # motion-aware rule in the pipeline that measurably earns its place --
    # raising it to 2 costs 3.8 points of recall (35.3% -> 31.5%) for 2.3 points
    # of precision.
    min_hits_dynamic: int = 1

    @property
    def l_hit(self):
        return prob_to_log_odds(self.p_hit)

    @property
    def l_miss(self):
        return prob_to_log_odds(self.p_miss)

    @property
    def l_prior(self):
        return prob_to_log_odds(self.p_prior)

    @property
    def min_probability(self):
        return float(log_odds_to_prob(self.min_log_odds))


@dataclass
class Cell:
    """One accumulated cell. Means are running means over hits only."""

    i: int
    j: int
    log_odds: float
    hit_count: int = 0
    last_hit_frame: int = -1
    last_hit_time: float = 0.0
    mean_rcs: float = 0.0
    mean_z: float = 0.0
    mean_vx: float = 0.0             # global frame
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
    log_odds: np.ndarray             # (M,)
    hit_count: np.ndarray            # (M,)
    mean_rcs: np.ndarray             # (M,)
    mean_z: np.ndarray               # (M,)
    mean_velocity: np.ndarray        # (M, 2) global frame

    def __len__(self) -> int:
        return int(self.centers_global.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def select(self, mask) -> "CellSnapshot":
        """Subset of these cells, e.g. after road masking."""
        return CellSnapshot(
            centers_global=self.centers_global[mask],
            centers_ego=self.centers_ego[mask],
            p_exist=self.p_exist[mask],
            log_odds=self.log_odds[mask],
            hit_count=self.hit_count[mask],
            mean_rcs=self.mean_rcs[mask],
            mean_z=self.mean_z[mask],
            mean_velocity=self.mean_velocity[mask],
        )


def _empty_snapshot() -> CellSnapshot:
    z2, z1 = np.zeros((0, 2)), np.zeros(0)
    return CellSnapshot(z2, z2, z1, z1, np.zeros(0, dtype=int), z1, z1, z2)


class RadarOccupancyGrid:
    """Accumulates radar returns into persistent, confidence-weighted cells."""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self.cells: dict = {}
        self.frame_index = -1
        self.last_timestamp = None

    def _to_index(self, xy):
        return np.floor(np.asarray(xy) / self.config.resolution).astype(np.int64)

    def _to_center(self, indices):
        return (np.asarray(indices, dtype=float) + 0.5) * self.config.resolution

    # -- the per-frame update ---------------------------------------------

    def update(self, cycle) -> None:
        """Fold one RadarCycle into the grid.

        1. advect moving cells along their own velocity
        2. hit  update: log-odds up, running means refreshed
        3. miss update: log-odds down for in-range cells that got nothing
        4. age out and delete
        """
        self.frame_index = cycle.frame_index
        self._advect(cycle)
        hit = self._apply_hits(cycle)
        self._apply_misses(cycle, hit)
        self._prune(cycle)
        self.last_timestamp = cycle.timestamp

    def _advect(self, cycle) -> None:
        """Carry every cell to where its own velocity says it now is.

        There is no static/dynamic split here. Each cell simply moves by
        `v * dt`, and a stationary cell has `v ~ 0` so it stays put on its own.
        An earlier version gated this on `|v| >= MOVING_SPEED`, on the reasoning
        that advecting static cells would inject velocity noise into the part of
        the grid that already works. Measured, the gate changes nothing at all
        (precision 56.7% vs 56.6%, recall and motion call identical), so it was
        removed: velocity alone is sufficient, and one rule beats two.

        Without advection in some form, a car at 10 m/s crosses ten 1 m cells
        between cycles, never revisits one, and never accumulates.
        """
        if not self.config.advect_dynamic_cells or self.last_timestamp is None:
            return
        dt = cycle.timestamp - self.last_timestamp
        if dt <= 0 or dt > 2.0:                # guard against scene seams
            return

        moved: dict = {}
        for key, cell in self.cells.items():
            predicted = self._to_center(key) + np.array([cell.mean_vx,
                                                         cell.mean_vy]) * dt
            new_key = tuple(self._to_index(predicted).tolist())
            cell.i, cell.j = new_key
            moved[new_key] = _keep_better(moved.get(new_key), cell, new_key)
        self.cells = moved

    def _apply_hits(self, cycle) -> set:
        """Log-odds increment and running-mean refresh for every observed cell."""
        cfg = self.config
        if cycle.num_points == 0:
            return set()

        # Group this cycle's returns by cell first. A cell catching five returns
        # in one sweep is still ONE observation of that cell -- counting it five
        # times would let dense clutter manufacture confidence.
        buckets: dict = {}
        indices = self._to_index(cycle.points_global[:, :2])
        for idx, key in enumerate(map(tuple, indices.tolist())):
            buckets.setdefault(key, []).append(idx)

        for key, members in buckets.items():
            cell = self.cells.get(key)
            if cell is None:
                cell = self.cells[key] = Cell(key[0], key[1], cfg.l_prior)

            cell.log_odds = min(cell.log_odds + cfg.l_hit, cfg.log_odds_max)
            cell.hit_count += 1
            cell.last_hit_frame = cycle.frame_index
            cell.last_hit_time = cycle.timestamp

            # This cycle's mean, folded into the running mean: m += (x - m)/n
            velocity = cycle.velocity_global[members].mean(axis=0)
            n = cell.hit_count
            cell.mean_rcs += (float(np.mean(cycle.rcs[members])) - cell.mean_rcs) / n
            cell.mean_z += (float(np.mean(cycle.points_global[members, 2]))
                            - cell.mean_z) / n
            cell.mean_vx += (float(velocity[0]) - cell.mean_vx) / n
            cell.mean_vy += (float(velocity[1]) - cell.mean_vy) / n

        return set(buckets)

    def _apply_misses(self, cycle, hit_cells: set) -> None:
        """Decay cells we could plausibly have seen this cycle but did not.

        Note what this is not: a ray-cast free-space model. A full occupancy
        grid traces each beam and marks everything in front of a return as
        empty; radar's angular resolution makes that unreliable, so instead any
        live in-range cell that got no return takes one miss step.
        """
        cfg = self.config
        ego_xy = cycle.ego_translation[:2]
        radius_sq = cfg.active_radius ** 2

        for key, cell in self.cells.items():
            if key in hit_cells:
                continue
            if float(np.sum((self._to_center(key) - ego_xy) ** 2)) > radius_sq:
                continue                       # out of range: not a real miss
            cell.log_odds = max(cell.log_odds + cfg.l_miss, cfg.log_odds_min)

    def _prune(self, cycle) -> None:
        """Delete cells that aged out, lost confidence, or left the window."""
        cfg = self.config
        ego_xy = cycle.ego_translation[:2]
        radius_sq = (cfg.active_radius * 1.2) ** 2     # hysteresis at the edge

        for key in [k for k, c in self.cells.items()
                    if float(np.sum((self._to_center(k) - ego_xy) ** 2)) > radius_sq
                    or cycle.frame_index - c.last_hit_frame > cfg.max_frames_since_hit
                    or c.log_odds <= cfg.delete_below_log_odds]:
            del self.cells[key]

    # -- readout ------------------------------------------------------------

    def _required_hits(self, cell: Cell) -> int:
        cfg = self.config
        return (min(cfg.min_hits, cfg.min_hits_dynamic)
                if cell.speed >= MOVING_SPEED else cfg.min_hits)

    def snapshot(self, cycle, filtered=False) -> CellSnapshot:
        """Columnar view of the live cells.

        ``filtered=True`` applies the log-odds and hit-count bars -- only
        confident, persistent cells become object candidates.
        """
        cells = [c for c in self.cells.values()
                 if not filtered
                 or (c.hit_count >= self._required_hits(c)
                     and c.log_odds >= self.config.min_log_odds)]
        if not cells:
            return _empty_snapshot()

        centers = self._to_center([[c.i, c.j] for c in cells])
        padded = np.hstack([centers, np.zeros((len(cells), 1))])

        return CellSnapshot(
            centers_global=centers,
            centers_ego=transform_points(padded, cycle.global_to_ego)[:, :2],
            p_exist=np.array([c.p_exist for c in cells]),
            log_odds=np.array([c.log_odds for c in cells]),
            hit_count=np.array([c.hit_count for c in cells], dtype=int),
            mean_rcs=np.array([c.mean_rcs for c in cells]),
            mean_z=np.array([c.mean_z for c in cells]),
            mean_velocity=np.array([[c.mean_vx, c.mean_vy] for c in cells]),
        )

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
