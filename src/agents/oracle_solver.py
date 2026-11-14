import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .csg import CADModel, CutoutAction
from .fea import FEASolver
from .problem import DesignProblem
from .simp import SIMPOptimizer


class ParametricCADOptimizer:
    """Converts optimal SIMP truss voids into clean, low-vertex filleted CSG CAD primitives."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.solver = FEASolver(problem)

    @staticmethod
    def _simplify_to_clean_polygon(
        poly: Polygon, cell_size: float, max_vertices: int = 8
    ) -> list[tuple[float, float]]:
        """Reduces a rasterized SIMP cavity to a clean <=max_vertices CAD polygon."""
        tol = cell_size * 1.15
        simplified = poly.simplify(tol, preserve_topology=True)
        for mult in (1.6, 2.2, 3.0, 4.0):
            if (
                isinstance(simplified, Polygon)
                and not simplified.is_empty
                and len(simplified.exterior.coords) - 1 <= max_vertices
            ):
                break
            simplified = poly.simplify(cell_size * mult, preserve_topology=True)

        if not isinstance(simplified, Polygon) or simplified.is_empty:
            return []

        coords = [
            (round(float(c[0]), 2), round(float(c[1]), 2))
            for c in list(simplified.exterior.coords)[:-1]
        ]
        if len(coords) > max_vertices:
            hull = simplified.convex_hull.simplify(
                cell_size * 1.8, preserve_topology=True
            )
            if isinstance(hull, Polygon) and not hull.is_empty:
                coords = [
                    (round(float(c[0]), 2), round(float(c[1]), 2))
                    for c in list(hull.exterior.coords)[:-1]
                ]
        if len(coords) < 3 or len(coords) > max_vertices:
            return []
        return coords

    def _fit_simp_void_polygons(
        self, model: CADModel, simp_density: np.ndarray
    ) -> None:
        p = self.problem
        dx = p.width / p.nx
        dy = p.height / p.ny
        cell_size = min(dx, dy)

        stock_grid = model.to_density_grid() > 0.5
        best_actions: list[CutoutAction] = []
        best_score = -np.inf

        for threshold in (0.32, 0.42, 0.50, 0.58):
            model.reset()
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
                merged = unary_union(boxes).buffer(cell_size * 0.4, quad_segs=8)
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
                    coords = self._simplify_to_clean_polygon(
                        poly, cell_size, max_vertices=8
                    )
                    if not coords:
                        continue

                    fillet_r = round(
                        max(p.min_feature_radius * 1.1, p.min_wall_thickness * 0.9),
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
                    if not wall_ok and sliver > 0.04:
                        model.undo()

            if not model.history or not model.is_structurally_connected():
                continue

            fea_res = self.solver.solve(model.to_density_grid(subgrid=2), penalty=1.0)
            vol = model.get_volume_fraction()
            vol_pen = max(0.0, vol - p.target_volume_fraction) * 6.0
            score = -fea_res.compliance * (1.0 + vol_pen)
            if score > best_score:
                best_score = score
                best_actions = list(model.history)

        model.reset()
        for act in best_actions:
            model.history.append(act)
        model._rebuild_geometry()

    def _expand_polygons_to_budget(self, model: CADModel) -> None:
        """Smoothly expands existing filleted polygon pockets to hit target volume without small circles."""
        p = self.problem
        cell_size = min(p.width / p.nx, p.height / p.ny)

        for _ in range(16):
            curr_vol = model.get_volume_fraction()
            if curr_vol <= p.target_volume_fraction + 0.004:
                break

            base_fea = self.solver.solve(model.to_density_grid(subgrid=2), penalty=1.0)
            base_comp = base_fea.compliance

            best_idx: int | None = None
            best_action: CutoutAction | None = None
            best_eff = np.inf

            for idx, orig_act in enumerate(list(model.history)):
                if orig_act.action_type != "polygon" or len(orig_act.params) < 7:
                    continue
                pts = [
                    (orig_act.params[i], orig_act.params[i + 1])
                    for i in range(0, len(orig_act.params) - 1, 2)
                ]
                fillet_r = float(orig_act.params[-1])
                orig_poly = Polygon(pts)

                for delta in (cell_size * 0.8, cell_size * 0.45, cell_size * 0.25):
                    buffered = orig_poly.buffer(
                        delta, join_style="mitre", mitre_limit=2.0
                    )
                    if not isinstance(buffered, Polygon) or buffered.is_empty:
                        continue
                    coords = self._simplify_to_clean_polygon(
                        buffered, cell_size * 0.4, max_vertices=8
                    )
                    if not coords:
                        continue

                    flat_coords: list[float] = []
                    for px, py in coords:
                        flat_coords.extend([px, py])
                    flat_coords.append(fillet_r)
                    cand_act = CutoutAction(
                        action_type="polygon",
                        params=tuple(flat_coords),
                        description=f"Cut {len(coords)}-gon pocket (fillet={fillet_r:.1f})",
                        mirror_x=orig_act.mirror_x,
                        mirror_y=orig_act.mirror_y,
                    )

                    model.history[idx] = cand_act
                    model._rebuild_geometry()
                    new_vol = model.get_volume_fraction()

                    valid = (
                        new_vol < curr_vol - 0.0005
                        and new_vol >= p.target_volume_fraction - 0.006
                        and not model.check_feature_constraints()
                        and model.is_structurally_connected()
                    )
                    if valid:
                        wall_ok, sliver = model.check_wall_thickness()
                        valid = wall_ok or sliver <= 0.04

                    if valid:
                        cand_fea = self.solver.solve(
                            model.to_density_grid(subgrid=2), penalty=1.0
                        )
                        cand_vm = float(np.max(cand_fea.von_mises_stress))
                        stress_ok = (
                            p.max_stress_limit is None
                            or cand_vm <= p.max_stress_limit * 1.02
                        )
                        if stress_ok and cand_fea.compliance <= base_comp * 1.2:
                            removed = curr_vol - new_vol
                            eff = (cand_fea.compliance - base_comp) / max(removed, 1e-6)
                            if eff < best_eff:
                                best_eff = eff
                                best_idx = idx
                                best_action = cand_act

                    model.history[idx] = orig_act
                    model._rebuild_geometry()

            if best_idx is None or best_action is None:
                break
            model.history[best_idx] = best_action
            model._rebuild_geometry()

    def _trim_to_budget(self, model: CADModel, simp_density: np.ndarray) -> None:
        p = self.problem
        dx = p.width / p.nx
        dy = p.height / p.ny
        xs = (np.arange(p.nx) + 0.5) * dx
        ys = (np.arange(p.ny) + 0.5) * dy

        while (
            model.get_volume_fraction() > p.target_volume_fraction + 0.004
            and len(model.history) < p.max_cutouts
        ):
            curr_vol = model.get_volume_fraction()
            density = model.to_density_grid(subgrid=1)
            base_fea = self.solver.solve(density, penalty=1.0)
            base_comp = base_fea.compliance

            solid = density > 0.75
            if not np.any(solid):
                break

            padded_solid = np.pad(solid, 1, mode="constant", constant_values=False)
            dist_phys = np.asarray(distance_transform_edt(padded_solid), dtype=float)[
                1:-1, 1:-1
            ] * min(dx, dy)
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
                p.min_feature_radius + 0.1,
                float(np.sqrt(rem_area / np.pi)) * 1.05,
            )

            placed = False
            for idx in order[:40]:
                if not np.isfinite(low_simp_score.flatten()[idx]):
                    break
                cx = round(float(xs[idx % p.nx]), 2)
                cy = round(float(ys[idx // p.nx]), 2)
                for r_try in sorted(
                    {
                        round(min(ideal_r, min(p.width, p.height) * 0.2), 2),
                        round(p.min_feature_radius + 0.4, 2),
                        round(p.min_feature_radius + 0.1, 2),
                    },
                    reverse=True,
                ):
                    if r_try < p.min_feature_radius + 0.2:
                        continue
                    cx_idx = idx % p.nx
                    cy_idx = idx // p.nx
                    min_clearance = r_try + max(p.min_wall_thickness * 0.5, 0.5)
                    if dist_phys[cy_idx, cx_idx] < min_clearance:
                        continue
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
                    cand_fea = self.solver.solve(
                        model.to_density_grid(subgrid=2), penalty=1.0
                    )
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
        self._expand_polygons_to_budget(model)
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
