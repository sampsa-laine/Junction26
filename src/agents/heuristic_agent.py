import numpy as np

from .csg import CADModel
from .fea import FEASolver
from .problem import DesignProblem, KeepOutType


class HeuristicCADDesigner:
    """Dedicated algorithmic designer that places CAD cutouts in regions of lowest strain energy."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.solver = FEASolver(problem)

    def design(
        self,
        max_steps: int = 15,
        cut_radius: float = 8.0,
        volume_tolerance: float = 0.02,
    ) -> CADModel:
        """Iteratively cuts material from lowest strain regions until volume target is reached."""
        model = CADModel(self.problem)
        target_vol = self.problem.target_volume_fraction
        dx = self.problem.width / self.problem.nx
        dy = self.problem.height / self.problem.ny

        xs = (np.arange(self.problem.nx) + 0.5) * dx
        ys = (np.arange(self.problem.ny) + 0.5) * dy
        x_grid, y_grid = np.meshgrid(xs, ys)

        # Precompute keepout & boundary-condition exclusion mask so zero-strain fixed flanges don't hog candidates
        keepout_mask = np.zeros((self.problem.ny, self.problem.nx), dtype=bool)
        for ko in self.problem.keepouts:
            if ko.keepout_type != KeepOutType.SOLID:
                continue
            if ko.shape == "circle":
                cx, cy, kr = ko.params
                keepout_mask |= np.hypot(x_grid - cx, y_grid - cy) <= kr + 1.0
            elif ko.shape == "rect":
                rx, ry, rw, rh = ko.params
                keepout_mask |= (
                    (x_grid >= rx - 1.0)
                    & (x_grid <= rx + rw + 1.0)
                    & (y_grid >= ry - 1.0)
                    & (y_grid <= ry + rh + 1.0)
                )

        min_r = max(1.5, self.problem.min_feature_radius + 0.05)

        for _ in range(min(max_steps, self.problem.max_cutouts)):
            current_vol = model.get_volume_fraction()
            if current_vol <= target_vol + volume_tolerance:
                break

            density = model.to_density_grid()
            fea_res = self.solver.solve(density, penalty=1.0)
            strain_energy = fea_res.element_strain_energies

            valid_mask = (density > 0.5) & (~keepout_mask)
            if not np.any(valid_mask):
                break

            masked_energy = np.where(valid_mask, strain_energy, np.inf)
            flat_indices = np.argsort(masked_energy.flatten())

            placed = False
            for idx in flat_indices[:60]:
                if not np.isfinite(masked_energy.flatten()[idx]):
                    break
                j = int(idx // self.problem.nx)
                i = int(idx % self.problem.nx)
                cand_x = float(xs[i])
                cand_y = float(ys[j])

                remaining_vol = current_vol - target_vol
                domain_area = self.problem.width * self.problem.height
                max_cut_area = remaining_vol * domain_area
                r = min(cut_radius, float(np.sqrt(max_cut_area / np.pi) * 1.1))
                r = max(min_r, r)

                near_support = any(
                    np.hypot(cand_x - s.x, cand_y - s.y) < max(r * 0.9, 3.0)
                    for s in self.problem.supports
                )
                near_load = any(
                    np.hypot(cand_x - l.x, cand_y - l.y) < max(r * 0.9, 3.5)
                    for l in self.problem.all_loads
                )
                if near_support or near_load:
                    continue

                if model.cut_circle(cand_x, cand_y, r):
                    if not model.is_structurally_connected():
                        model.undo()
                        continue
                    test_fea = self.solver.solve(model.to_density_grid(), penalty=1.0)
                    if test_fea.compliance > fea_res.compliance * 8.0:
                        model.undo()
                        continue
                    placed = True
                    break

            if not placed:
                break

        return model
