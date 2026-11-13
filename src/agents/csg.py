from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, box, LineString
from shapely.ops import unary_union

from .problem import DesignProblem, KeepOutType


@dataclass
class CutoutAction:
    action_type: str  # "circle", "rect", "slot"
    params: tuple[float, ...]
    description: str = ""


class CADModel:
    """Parametric Constructive Solid Geometry (CSG) 2D CAD component model."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.width = problem.width
        self.height = problem.height
        self.stock_plate: Polygon = box(0.0, 0.0, self.width, self.height)
        self.history: list[CutoutAction] = []

        # Keep-out zones
        self._build_keepouts()
        self._rebuild_geometry()

    def _build_keepouts(self) -> None:
        solid_geoms = []
        void_geoms = []
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

        self.solid_keepouts = unary_union(solid_geoms) if solid_geoms else Polygon()
        self.void_keepouts = unary_union(void_geoms) if void_geoms else Polygon()

    def cut_circle(self, x: float, y: float, radius: float) -> bool:
        """Drills a circular hole, respecting solid keepouts."""
        if radius <= 0.0:
            return False
        hole = Point(x, y).buffer(radius, quad_segs=16)
        # Cannot cut into solid keepouts
        if not self.solid_keepouts.is_empty and hole.intersects(self.solid_keepouts):
            hole = hole.difference(self.solid_keepouts)
            if hole.is_empty:
                return False

        action = CutoutAction(
            action_type="circle",
            params=(x, y, radius),
            description=f"Cut circle at ({x:.1f}, {y:.1f}) r={radius:.1f}",
        )
        self.history.append(action)
        self._rebuild_geometry()
        return True

    def cut_rect(self, x: float, y: float, w: float, h: float) -> bool:
        """Punches a rectangular cutout, respecting solid keepouts."""
        if w <= 0.0 or h <= 0.0:
            return False
        cut = box(x, y, x + w, y + h)
        if not self.solid_keepouts.is_empty and cut.intersects(self.solid_keepouts):
            cut = cut.difference(self.solid_keepouts)
            if cut.is_empty:
                return False

        action = CutoutAction(
            action_type="rect",
            params=(x, y, w, h),
            description=f"Cut rect at ({x:.1f}, {y:.1f}) {w:.1f}x{h:.1f}",
        )
        self.history.append(action)
        self._rebuild_geometry()
        return True

    def cut_slot(
        self, x1: float, y1: float, x2: float, y2: float, radius: float
    ) -> bool:
        """Cuts an elongated rounded slot along a line segment."""
        if radius <= 0.0 or (x1 == x2 and y1 == y2):
            return False
        line = LineString([(x1, y1), (x2, y2)])
        cut = line.buffer(radius, quad_segs=16)
        if not self.solid_keepouts.is_empty and cut.intersects(self.solid_keepouts):
            cut = cut.difference(self.solid_keepouts)
            if cut.is_empty:
                return False

        action = CutoutAction(
            action_type="slot",
            params=(x1, y1, x2, y2, radius),
            description=f"Cut slot from ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f}) r={radius:.1f}",
        )
        self.history.append(action)
        self._rebuild_geometry()
        return True

    def undo(self) -> bool:
        """Reverts the most recent cutout action."""
        if not self.history:
            return False
        self.history.pop()
        self._rebuild_geometry()
        return True

    def _rebuild_geometry(self) -> None:
        """Reconstructs the 2D polygon by applying all cutouts."""
        geom = self.stock_plate

        # Cut void keepouts first
        if not self.void_keepouts.is_empty:
            geom = geom.difference(self.void_keepouts)

        # Apply cutout actions
        cut_geoms = []
        for action in self.history:
            if action.action_type == "circle":
                cx, cy, r = action.params
                cut_geoms.append(Point(cx, cy).buffer(r, quad_segs=16))
            elif action.action_type == "rect":
                rx, ry, rw, rh = action.params
                cut_geoms.append(box(rx, ry, rx + rw, ry + rh))
            elif action.action_type == "slot":
                x1, y1, x2, y2, r = action.params
                cut_geoms.append(
                    LineString([(x1, y1), (x2, y2)]).buffer(r, quad_segs=16)
                )

        if cut_geoms:
            all_cuts = unary_union(cut_geoms)
            if not self.solid_keepouts.is_empty:
                all_cuts = all_cuts.difference(self.solid_keepouts)
            geom = geom.difference(all_cuts)

        self.current_geometry = geom

    def get_volume_fraction(self) -> float:
        """Computes current solid area fraction relative to domain."""
        total_area = self.width * self.height
        return float(self.current_geometry.area / total_area)

    def to_density_grid(
        self, nx: int | None = None, ny: int | None = None, subgrid: int = 3
    ) -> np.ndarray:
        """Rasterizes the CAD geometry into an (ny, nx) element density matrix.

        Uses sub-sampling (subgrid x subgrid points per element) for fast, smooth area fraction.
        """
        nx = nx or self.problem.nx
        ny = ny or self.problem.ny
        dx = self.width / nx
        dy = self.height / ny

        # Subgrid offsets within each element
        sub_offsets = (np.arange(subgrid) + 0.5) / subgrid

        densities = np.zeros((ny, nx), dtype=float)
        # Vectorized point testing via Shapely prepared geometry
        from shapely.prepared import prep

        prepared_geom = prep(self.current_geometry)

        for j in range(ny):
            y_base = j * dy
            for i in range(nx):
                x_base = i * dx
                # Sample subgrid points
                inside_count = 0
                for sy in sub_offsets:
                    py = y_base + sy * dy
                    for sx in sub_offsets:
                        px = x_base + sx * dx
                        if prepared_geom.contains(Point(px, py)):
                            inside_count += 1
                densities[j, i] = inside_count / (subgrid * subgrid)

        return densities

    def to_svg(self) -> str:
        """Generates an SVG representation of the CAD component."""
        # Viewbox with margin
        margin = 5
        vb_w = self.width + 2 * margin
        vb_h = self.height + 2 * margin

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-margin} {-margin} {vb_w} {vb_h}" width="100%" height="100%">',
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#f0f0f0" stroke="#ccc" stroke-dasharray="2,2"/>',
        ]

        # Draw solid geometry
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

        geom = self.current_geometry
        if isinstance(geom, Polygon):
            d_path = polygon_to_path(geom)
            svg_parts.append(
                f'<path d="{d_path}" fill="#3b82f6" fill-rule="evenodd" stroke="#1d4ed8" stroke-width="0.5"/>'
            )
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                d_path = polygon_to_path(poly)
                svg_parts.append(
                    f'<path d="{d_path}" fill="#3b82f6" fill-rule="evenodd" stroke="#1d4ed8" stroke-width="0.5"/>'
                )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
