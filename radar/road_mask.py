"""Drop grid cells that fall outside the drivable area, using NuScenesMap.

Radar returns a great deal of off-road structure -- barriers, walls, fences,
vegetation -- none of which nuScenes annotates as a detection. Discarding cells
outside the drivable surface removes a large source of false positives before
clustering.

**This also introduces false negatives, and that is not a side note.** A
pedestrian on a sidewalk, a cyclist on a bike path, a car parked half on the
kerb: all sit outside `drivable_area` and are all thrown away here. The filter
raises precision by lowering recall on anything off the road surface. It is a
defensible simplification for a ground-vehicle-focused baseline and nothing
more; the README states it plainly.

Requires the **nuScenes map expansion pack**, which is a separate download from
the mini/trainval data. Without it this module disables itself and says so
rather than failing, so the rest of the pipeline still runs.
"""

from __future__ import annotations

import os

import numpy as np

#: Map layers counted as "road". `drivable_area` is the union of road surfaces;
#: `road_segment` and `lane` are narrower and are included so that cells on a
#: lane not covered by a drivable_area polygon still survive.
ROAD_LAYERS = ("drivable_area", "road_segment", "lane")


class RoadMask:
    """Point-in-polygon test against a scene's drivable area.

    Polygons are loaded once per map location and indexed with an STRtree, so
    masking a few thousand cells per frame is a cheap query rather than a scan
    over every polygon in the city.
    """

    def __init__(self, dataroot: str, location: str, layers=ROAD_LAYERS):
        self.location = location
        self.available = False
        self.reason = ""
        self._tree = None
        self._polygons = []

        expansion = os.path.join(dataroot, "maps", "expansion", f"{location}.json")
        if not os.path.exists(expansion):
            self.reason = (
                f"map expansion not found at maps/expansion/{location}.json -- "
                f"download the nuScenes map expansion pack to enable road masking")
            return

        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
            from shapely.strtree import STRtree
        except ImportError as error:
            self.reason = f"map dependencies unavailable: {error}"
            return

        nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)
        for layer in layers:
            if layer not in nusc_map.non_geometric_polygon_layers:
                continue
            for record in getattr(nusc_map, layer):
                for token in record.get("polygon_tokens", [record.get("polygon_token")]):
                    if token is None:
                        continue
                    polygon = nusc_map.extract_polygon(token)
                    if polygon.is_valid and not polygon.is_empty:
                        self._polygons.append(polygon)

        if not self._polygons:
            self.reason = f"no road polygons found in {location}"
            return

        self._tree = STRtree(self._polygons)
        self.available = True

    def contains(self, points_global: np.ndarray) -> np.ndarray:
        """(M,) bool: does each global-frame (x, y) lie on the road surface?

        Returns all-True when the map is unavailable, so a missing expansion
        pack degrades to "no masking" rather than silently dropping everything.
        """
        points_global = np.asarray(points_global, dtype=float)
        if not self.available or points_global.shape[0] == 0:
            return np.ones(points_global.shape[0], dtype=bool)

        from shapely.geometry import Point

        inside = np.zeros(points_global.shape[0], dtype=bool)
        for index, (x, y) in enumerate(points_global[:, :2]):
            point = Point(float(x), float(y))
            for candidate in self._tree.query(point):
                polygon = (self._polygons[candidate]
                           if isinstance(candidate, (int, np.integer)) else candidate)
                if polygon.contains(point):
                    inside[index] = True
                    break
        return inside

    def apply(self, snapshot):
        """Filter a CellSnapshot down to cells on the drivable surface."""
        if not self.available or snapshot.is_empty:
            return snapshot, 0
        keep = self.contains(snapshot.centers_global)
        return snapshot.select(keep), int((~keep).sum())


def for_scene(nusc, scene_token: str, enabled=True) -> RoadMask | None:
    """Build a RoadMask for the map location this scene was recorded in."""
    if not enabled:
        return None
    scene = nusc.get("scene", scene_token)
    location = nusc.get("log", scene["log_token"])["location"]
    return RoadMask(nusc.dataroot, location)
