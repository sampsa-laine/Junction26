from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .problem import DesignProblem, KeepOutType


@dataclass
class CutoutAction:
    action_type: str  # "circle", "rect", "slot", "polygon"
    params: tuple[float, ...]
    description: str = ""
    mirror_x: bool = False
    mirror_y: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "params": [round(float(v), 4) for v in self.params],
            "description": self.description,
            "mirror_x": self.mirror_x,
            "mirror_y": self.mirror_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CutoutAction":
        return cls(
            action_type=str(data["action_type"]),
            params=tuple(float(v) for v in data["params"]),
            description=str(data.get("description", "")),
            mirror_x=bool(data.get("mirror_x", False)),
            mirror_y=bool(data.get("mirror_y", False)),
        )


class CADModel:
    """Parametric Constructive Solid Geometry (CSG) 2D CAD component model."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.width = problem.width
        self.height = problem.height
        self.stock_plate: Polygon = box(0.0, 0.0, self.width, self.height)
        self.history: list[CutoutAction] = []
        self.current_geometry: BaseGeometry = self.stock_plate

        self._build_keepouts()
        self._rebuild_geometry()

    def _build_keepouts(self) -> None:
        solid_geoms: list[BaseGeometry] = []
        void_geoms: list[BaseGeometry] = []
        for ko in self.problem.keepouts:
            if ko.shape == "circle":
                cx, cy, r = ko.params
                geom = Point(cx, cy).buffer(r, quad_segs=16)
            elif ko.shape == "rect":
                rx, ry, rw, rh = ko.params
                geom = box(rx, ry, rx + rw, ry + rh)
            else:
                continue

            if ko.keepout_type == KeepOutType.SOLID:
                solid_geoms.append(geom)
            else:
                void_geoms.append(geom)

        self.solid_keepouts: BaseGeometry = (
            unary_union(solid_geoms) if solid_geoms else Polygon()
        )
        self.void_keepouts: BaseGeometry = (
            unary_union(void_geoms) if void_geoms else Polygon()
        )

    def _apply_mirrors(
        self, geom: BaseGeometry, mirror_x: bool, mirror_y: bool
    ) -> BaseGeometry:
        parts = [geom]
        if mirror_x:
            parts.append(
                scale(geom, xfact=-1.0, yfact=1.0, origin=(self.width / 2.0, 0.0))
            )
        if mirror_y:
            current_parts = list(parts)
            for p in current_parts:
                parts.append(
                    scale(p, xfact=1.0, yfact=-1.0, origin=(0.0, self.height / 2.0))
                )
        return unary_union(parts) if len(parts) > 1 else geom

    def _action_to_geometry(self, action: CutoutAction) -> BaseGeometry:
        geom: BaseGeometry = Polygon()
        if action.action_type == "circle":
            cx, cy, r = action.params
            if r > 0.0:
                geom = Point(cx, cy).buffer(r, quad_segs=16)
        elif action.action_type == "rect":
            if len(action.params) == 4:
                rx, ry, rw, rh = action.params
                fillet = 0.0
            else:
                rx, ry, rw, rh, fillet = action.params
            if rw > 0.0 and rh > 0.0:
                max_f = min(rw, rh) * 0.49
                f = max(0.0, min(fillet, max_f))
                if f > 1e-6:
                    geom = box(rx + f, ry + f, rx + rw - f, ry + rh - f).buffer(
                        f, quad_segs=12
                    )
                else:
                    geom = box(rx, ry, rx + rw, ry + rh)
        elif action.action_type == "slot":
            x1, y1, x2, y2, r = action.params
            if r > 0.0 and (x1 != x2 or y1 != y2):
                geom = LineString([(x1, y1), (x2, y2)]).buffer(r, quad_segs=16)
        elif action.action_type == "polygon":
            # Params encoded as (x1, y1, x2, y2, ..., xn, yn, fillet_radius)
            if len(action.params) >= 7 and len(action.params) % 2 == 1:
                fillet = max(0.0, action.params[-1])
                coords = [
                    (action.params[k], action.params[k + 1])
                    for k in range(0, len(action.params) - 1, 2)
                ]
                raw_poly = Polygon(coords).buffer(0.0)
                if not raw_poly.is_empty:
                    if fillet > 1e-6:
                        eroded = raw_poly.buffer(-fillet, quad_segs=12)
                        geom = (
                            eroded.buffer(fillet, quad_segs=12)
                            if not eroded.is_empty
                            else raw_poly
                        )
                    else:
                        geom = raw_poly

        if geom.is_empty:
            return geom
        return self._apply_mirrors(geom, action.mirror_x, action.mirror_y)

    def _commit_cut(self, action: CutoutAction) -> bool:
        cut = self._action_to_geometry(action)
        if cut.is_empty:
            return False
        if not self.solid_keepouts.is_empty and cut.intersects(self.solid_keepouts):
            cut = cut.difference(self.solid_keepouts)
            if cut.is_empty:
                return False
        if not cut.intersects(self.current_geometry):
            return False

        self.history.append(action)
        self._rebuild_geometry()
        return True

    def cut_circle(
        self,
        x: float,
        y: float,
        radius: float,
        mirror_x: bool = False,
        mirror_y: bool = False,
    ) -> bool:
        """Drills a circular hole, respecting solid keepouts."""
        if radius <= 0.0:
            return False
        action = CutoutAction(
            action_type="circle",
            params=(x, y, radius),
            description=f"Cut circle at ({x:.1f}, {y:.1f}) r={radius:.1f}",
            mirror_x=mirror_x,
            mirror_y=mirror_y,
        )
        return self._commit_cut(action)

    def cut_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fillet_radius: float = 0.0,
        mirror_x: bool = False,
        mirror_y: bool = False,
    ) -> bool:
        """Punches a rectangular or rounded-rectangle cutout, respecting solid keepouts."""
        if w <= 0.0 or h <= 0.0:
            return False
        params = (x, y, w, h, fillet_radius) if fillet_radius > 0.0 else (x, y, w, h)
        fillet_str = f" r={fillet_radius:.1f}" if fillet_radius > 0.0 else ""
        action = CutoutAction(
            action_type="rect",
            params=params,
            description=f"Cut rect at ({x:.1f}, {y:.1f}) {w:.1f}x{h:.1f}{fillet_str}",
            mirror_x=mirror_x,
            mirror_y=mirror_y,
        )
        return self._commit_cut(action)

    def cut_slot(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
        mirror_x: bool = False,
        mirror_y: bool = False,
    ) -> bool:
        """Cuts an elongated rounded slot along a line segment."""
        if radius <= 0.0 or (x1 == x2 and y1 == y2):
            return False
        action = CutoutAction(
            action_type="slot",
            params=(x1, y1, x2, y2, radius),
            description=f"Cut slot ({x1:.1f}, {y1:.1f})->({x2:.1f}, {y2:.1f}) r={radius:.1f}",
            mirror_x=mirror_x,
            mirror_y=mirror_y,
        )
        return self._commit_cut(action)

    def cut_polygon(
        self,
        points: list[tuple[float, float]],
        fillet_radius: float = 0.0,
        mirror_x: bool = False,
        mirror_y: bool = False,
    ) -> bool:
        """Cuts an arbitrary polygonal truss pocket with optional corner filleting."""
        if len(points) < 3:
            return False
        flat_coords: list[float] = []
        for px, py in points:
            flat_coords.extend([float(px), float(py)])
        flat_coords.append(max(0.0, float(fillet_radius)))
        action = CutoutAction(
            action_type="polygon",
            params=tuple(flat_coords),
            description=f"Cut {len(points)}-gon pocket (fillet={fillet_radius:.1f})",
            mirror_x=mirror_x,
            mirror_y=mirror_y,
        )
        return self._commit_cut(action)

    def undo(self) -> bool:
        """Reverts the most recent cutout action."""
        if not self.history:
            return False
        self.history.pop()
        self._rebuild_geometry()
        return True

    def reset(self) -> None:
        """Clears all cutouts back to the initial stock plate."""
        self.history.clear()
        self._rebuild_geometry()

    def _rebuild_geometry(self) -> None:
        geom: BaseGeometry = self.stock_plate

        if not self.void_keepouts.is_empty:
            geom = geom.difference(self.void_keepouts)

        cut_geoms = [self._action_to_geometry(a) for a in self.history]
        cut_geoms = [g for g in cut_geoms if not g.is_empty]

        if cut_geoms:
            all_cuts = unary_union(cut_geoms)
            if not self.solid_keepouts.is_empty:
                all_cuts = all_cuts.difference(self.solid_keepouts)
            geom = geom.difference(all_cuts)

        self.current_geometry = geom.buffer(0.0) if not geom.is_empty else Polygon()

    def get_stock_volume_fraction(self) -> float:
        """Computes initial solid area fraction before any user cutouts."""
        geom: BaseGeometry = self.stock_plate
        if not self.void_keepouts.is_empty:
            geom = geom.difference(self.void_keepouts)
        return float(geom.area / (self.width * self.height))

    def get_volume_fraction(self) -> float:
        """Computes current solid area fraction relative to domain bounding box."""
        total_area = self.width * self.height
        return float(self.current_geometry.area / total_area)

    def to_density_grid(
        self, nx: int | None = None, ny: int | None = None, subgrid: int = 3
    ) -> np.ndarray:
        """Vectorized sub-grid rasterization into an (ny, nx) element density matrix."""
        nx = nx or self.problem.nx
        ny = ny or self.problem.ny
        if self.current_geometry.is_empty:
            return np.zeros((ny, nx), dtype=float)

        px = (np.arange(nx * subgrid) + 0.5) * (self.width / (nx * subgrid))
        py = (np.arange(ny * subgrid) + 0.5) * (self.height / (ny * subgrid))
        px_grid, py_grid = np.meshgrid(px, py)

        inside = shapely.contains_xy(self.current_geometry, px_grid, py_grid)
        return inside.reshape(ny, subgrid, nx, subgrid).mean(axis=(1, 3))

    def _extract_polygons(self, geom: BaseGeometry) -> list[Polygon]:
        if geom.is_empty:
            return []
        if isinstance(geom, Polygon):
            return [geom]
        if isinstance(geom, MultiPolygon):
            return list(geom.geoms)
        return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]

    def is_structurally_connected(self, geom: BaseGeometry | None = None) -> bool:
        """Checks whether a single contiguous solid body connects supports to all loads."""
        target_geom = self.current_geometry if geom is None else geom
        polys = self._extract_polygons(target_geom)
        if not polys:
            return False

        tol = max(self.width / self.problem.nx, self.height / self.problem.ny) * 1.25
        support_zone = unary_union(
            [
                Point(s.x, s.y).buffer(max(s.tolerance, tol))
                for s in self.problem.supports
            ]
        )
        load_pts = [Point(l.x, l.y).buffer(tol) for l in self.problem.all_loads]

        for poly in polys:
            if not poly.intersects(support_zone):
                continue
            if all(poly.intersects(lp) for lp in load_pts):
                # Reject large floating disconnected islands (>2% of solid area)
                main_ratio = poly.area / max(target_geom.area, 1e-9)
                return main_ratio >= 0.98
        return False

    def check_wall_thickness(
        self, min_thickness: float | None = None
    ) -> tuple[bool, float]:
        """Evaluates minimum wall thickness via morphological erosion/opening.

        Returns (passes_check, thin_wall_sliver_ratio).
        """
        min_t = (
            self.problem.min_wall_thickness if min_thickness is None else min_thickness
        )
        if min_t <= 0.0 or self.current_geometry.is_empty:
            return not self.current_geometry.is_empty, 0.0

        radius = 0.45 * min_t
        eroded = self.current_geometry.buffer(-radius, quad_segs=12)
        if eroded.is_empty:
            return False, 1.0

        opened = eroded.buffer(radius, quad_segs=12)
        stock_geom: BaseGeometry = self.stock_plate
        if not self.void_keepouts.is_empty:
            stock_geom = stock_geom.difference(self.void_keepouts)
        stock_corner_loss = max(
            0.0,
            stock_geom.area
            - stock_geom.buffer(-radius, quad_segs=12)
            .buffer(radius, quad_segs=12)
            .area,
        )

        excess_loss = max(
            0.0, (self.current_geometry.area - opened.area) - stock_corner_loss * 1.2
        )
        sliver_ratio = float(excess_loss / max(self.current_geometry.area, 1e-9))

        # Load-bearing paths should remain connected under mild erosion (0.30 * min_t)
        eroded_bridge = self.current_geometry.buffer(-0.30 * min_t, quad_segs=12)
        bridge_ok = self.is_structurally_connected(
            eroded_bridge.buffer(0.30 * min_t, quad_segs=12)
        )
        passes = bridge_ok and sliver_ratio <= 0.05
        return passes, sliver_ratio

    def check_boundary_smoothness(self) -> tuple[bool, float]:
        """Evaluates CNC manufacturability & smoothness of parametric cutouts.

        Penalizes unfilleted sharp polygon pockets and jagged high-vertex contours
        (>8 vertices), returning (is_smooth, smoothness_factor in [0.65, 1.0]).
        """
        if not self.history:
            return True, 1.0

        penalty = 0.0
        for action in self.history:
            if action.action_type == "polygon":
                n_verts = (len(action.params) - 1) // 2
                fillet = action.params[-1]
                if n_verts > 8:
                    penalty += 0.04 * (n_verts - 8)
                if fillet < self.problem.min_feature_radius * 0.5:
                    penalty += 0.06
            elif action.action_type == "rect":
                fillet = action.params[4] if len(action.params) >= 5 else 0.0
                if fillet < self.problem.min_feature_radius * 0.25:
                    penalty += 0.03

        factor = float(max(0.65, 1.0 - penalty))
        return factor >= 0.95, factor

    def check_feature_constraints(self) -> list[str]:
        """Validates cutout count and minimum tool/feature radius constraints."""
        violations: list[str] = []
        if len(self.history) > self.problem.max_cutouts:
            violations.append(
                f"Cutout count {len(self.history)} exceeds max_cutouts={self.problem.max_cutouts}"
            )

        min_r = self.problem.min_feature_radius - 1e-6
        for idx, action in enumerate(self.history):
            if action.action_type == "circle":
                if action.params[2] < min_r:
                    violations.append(
                        f"Action {idx + 1} circle radius {action.params[2]:.2f} < {self.problem.min_feature_radius:.2f}"
                    )
            elif action.action_type == "slot":
                if action.params[4] < min_r:
                    violations.append(
                        f"Action {idx + 1} slot radius {action.params[4]:.2f} < {self.problem.min_feature_radius:.2f}"
                    )
            elif action.action_type == "rect":
                half_min = 0.5 * min(action.params[2], action.params[3])
                if half_min < min_r:
                    violations.append(
                        f"Action {idx + 1} rect half-width {half_min:.2f} < {self.problem.min_feature_radius:.2f}"
                    )
            elif action.action_type == "polygon":
                poly = self._action_to_geometry(action)
                if poly.buffer(-min_r * 0.8).is_empty:
                    violations.append(
                        f"Action {idx + 1} polygon narrower than min feature radius {self.problem.min_feature_radius:.2f}"
                    )
        return violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_name": self.problem.name,
            "volume_fraction": round(self.get_volume_fraction(), 5),
            "actions": [a.to_dict() for a in self.history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], problem: DesignProblem) -> "CADModel":
        model = cls(problem)
        for item in data.get("actions", []):
            action = CutoutAction.from_dict(item)
            model.history.append(action)
        model._rebuild_geometry()
        return model

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path, problem: DesignProblem) -> "CADModel":
        return cls.from_dict(json.loads(Path(path).read_text()), problem)

    def to_svg(self) -> str:
        """Generates an SVG representation of the CAD component (y-axis pointing up)."""
        margin = 5.0
        vb_w = self.width + 2.0 * margin
        vb_h = self.height + 2.0 * margin

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-margin} {-margin} {vb_w} {vb_h}" width="100%" height="100%">',
            f'<g transform="translate(0, {self.height}) scale(1, -1)">',
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#f0f0f0" stroke="#ccc" stroke-dasharray="2,2"/>',
        ]

        def polygon_to_path(poly: Polygon) -> str:
            coords = list(poly.exterior.coords)
            d = f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"
            for x, y in coords[1:]:
                d += f" L {x:.2f} {y:.2f}"
            d += " Z"
            for interior in poly.interiors:
                icoords = list(interior.coords)
                d += f" M {icoords[0][0]:.2f} {icoords[0][1]:.2f}"
                for x, y in icoords[1:]:
                    d += f" L {x:.2f} {y:.2f}"
                d += " Z"
            return d

        for poly in self._extract_polygons(self.current_geometry):
            d_path = polygon_to_path(poly)
            svg_parts.append(
                f'<path d="{d_path}" fill="#3b82f6" fill-rule="evenodd" stroke="#1d4ed8" stroke-width="0.5"/>'
            )

        svg_parts.append("</g>")
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
