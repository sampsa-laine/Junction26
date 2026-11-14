import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .csg import CADModel
from .fea import FEASolver
from .problem import DesignProblem
from .simp import SIMPOptimizer


class ParametricCADOptimizer:
    """Converts optimal SIMP truss voids + FEA refinement into clean parametric CSG primitives."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.solver = FEASolver(problem)

    def _fit_simp_void_polygons(
        self, model: CADModel, simp_density: np.ndarray
    ) -> None:
        p = self.problem
        dx = p.width / p.nx
        dy = p.height / p.ny
        cell_size = min(dx, dy)

        stock_grid = model.to_density_grid() > 0.5

        for threshold in (0.30, 0.42, 0.50):
            if model.get_volume_fraction() <= p.target_volume_fraction + 0.002:
                break
            if len(model.history) >= p.max_cutouts - 2:
                break

            # Keep buffer around high-density SIMP struts so wall thickness stays healthy
            strut_mask = simp_density >= threshold
            strut_clearance = (
                np.asarray(distance_transform_edt(~strut_mask), dtype=float) * cell_size
            )
            void_mask = stock_grid & (
                strut_clearance >= max(p.min_wall_thickness * 0.35, cell_size * 0.45)
            )

            raw_labeled: Any = label(void_mask)
            labeled = np.asarray(raw_labeled[0], dtype=int)
            num_regions = int(raw_labeled[1])
            if num_regions == 0:
                continue

            regions = sorted(
                [
                    (int(np.sum(labeled == r_id)), r_id)
                    for r_id in range(1, num_regions + 1)
                ],
                reverse=True,
            )

            model.reset()
            for count, r_id in regions:
                if count < 5 or len(model.history) >= p.max_cutouts - 2:
                    continue
                if model.get_volume_fraction() <= p.target_volume_fraction - 0.003:
                    break

                ys_idx, xs_idx = np.where(labeled == r_id)
                boxes = [
                    box(
                        float(ix * dx),
                        float(iy * dy),
                        float((ix + 1) * dx),
                        float((iy + 1) * dy),
                    )
                    for iy, ix in zip(ys_idx, xs_idx)
                ]
                merged = unary_union(boxes).buffer(cell_size * 0.35, quad_segs=8)
                polys = (
                    [merged]
                    if isinstance(merged, Polygon)
                    else list(merged.geoms)
                    if isinstance(merged, MultiPolygon)
                    else []
                )
                for poly in sorted(polys, key=lambda g: g.area, reverse=True):
                    if poly.area < (p.min_feature_radius**2) * np.pi:
                        continue
                    simplified = poly.simplify(cell_size * 0.85, preserve_topology=True)
                    if not isinstance(simplified, Polygon) or simplified.is_empty:
                        continue
                    coords: list[tuple[float, float]] = [
                        (float(c[0]), float(c[1]))
                        for c in list(simplified.exterior.coords)[:-1]
                    ]
                    if len(coords) < 3:
                        continue

                    fillet_r = round(
                        max(p.min_feature_radius * 0.85, p.min_wall_thickness * 0.75),
                        2,
                    )
                    if not model.cut_polygon(coords, fillet_radius=fillet_r):
                        continue

                    if (
                        model.check_feature_constraints()
                        or not model.is_structurally_connected()
                    ):
                        model.undo()
                        continue

                    wall_ok, sliver = model.check_wall_thickness()
                    if not wall_ok and sliver > 0.045:
                        model.undo()

            if model.get_volume_fraction() <= p.target_volume_fraction + 0.03:
                break

    def _trim_to_budget(self, model: CADModel, simp_density: np.ndarray) -> None:
        p = self.problem
        dx = p.width / p.nx
        dy = p.height / p.ny
        xs = (np.arange(p.nx) + 0.5) * dx
        ys = (np.arange(p.ny) + 0.5) * dy

        while (
            model.get_volume_fraction() > p.target_volume_fraction + 0.001
            and len(model.history) < p.max_cutouts
        ):
            curr_vol = model.get_volume_fraction()
            density = model.to_density_grid()
            base_fea = self.solver.solve(density, penalty=1.0)
            base_comp = base_fea.compliance

            solid = density > 0.75
            if not np.any(solid):
                break

            # Prefer solid elements where SIMP density & strain energy are lowest
            dist_phys = np.asarray(distance_transform_edt(solid), dtype=float) * min(
                dx, dy
            )
            low_simp_score = np.where(
                solid,
                (1.0 - simp_density) + 0.08 * dist_phys,
                -np.inf,
            )
            order = np.argsort(low_simp_score.flatten())[::-1]

            rem_area = (
                max(0.002, curr_vol - p.target_volume_fraction) * p.width * p.height
            )
            ideal_r = max(
                p.min_feature_radius + 0.08,
                float(np.sqrt(rem_area / np.pi)) * 1.05,
            )

            placed = False
            for idx in order[:40]:
                if not np.isfinite(low_simp_score.flatten()[idx]):
                    break
                cx = float(xs[idx % p.nx])
                cy = float(ys[idx // p.nx])
                for r_try in sorted(
                    {
                        round(min(ideal_r, min(p.width, p.height) * 0.2), 2),
                        round(p.min_feature_radius + 0.35, 2),
                        round(p.min_feature_radius + 0.08, 2),
                    },
                    reverse=True,
                ):
                    if not model.cut_circle(cx, cy, r_try):
                        continue
                    new_vol = model.get_volume_fraction()
                    if (
                        curr_vol - new_vol < 0.002
                        or new_vol < p.target_volume_fraction - 0.025
                        or model.check_feature_constraints()
                        or not model.is_structurally_connected()
                    ):
                        model.undo()
                        continue

                    wall_ok, sliver = model.check_wall_thickness()
                    cand_fea = self.solver.solve(model.to_density_grid(), penalty=1.0)
                    cand_vm = float(np.max(cand_fea.von_mises_stress))
                    stress_ok = (
                        p.max_stress_limit is None
                        or cand_vm <= p.max_stress_limit * 1.02
                    )
                    if (
                        (wall_ok or sliver <= 0.04)
                        and stress_ok
                        and cand_fea.compliance <= base_comp * 1.25
                    ):
                        placed = True
                        break
                    model.undo()
                if placed:
                    break
            if not placed:
                break

    def optimize(self) -> CADModel:
        simp_res = SIMPOptimizer(self.problem).optimize(max_iterations=40)
        model = CADModel(self.problem)
        self._fit_simp_void_polygons(model, simp_res.optimal_density)
        self._trim_to_budget(model, simp_res.optimal_density)
        return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reference SIMP-to-CAD Parametric Oracle Solver"
    )
    parser.add_argument("--problem", default="/app/problem.json")
    parser.add_argument("--design", default="/app/design.json")
    args = parser.parse_args(argv)

    problem = DesignProblem.load_json(args.problem)
    optimizer = ParametricCADOptimizer(problem)
    model = optimizer.optimize()
    Path(args.design).parent.mkdir(parents=True, exist_ok=True)
    model.save_json(args.design)
    print(
        f"Saved reference design with {len(model.history)} actions "
        f"(volume_fraction={model.get_volume_fraction():.4f}) to {args.design}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
