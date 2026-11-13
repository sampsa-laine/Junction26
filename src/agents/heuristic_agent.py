import numpy as np

from .csg import CADModel
from .fea import FEASolver
from .problem import DesignProblem


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

        # Element centers
        xs = (np.arange(self.problem.nx) + 0.5) * dx
        ys = (np.arange(self.problem.ny) + 0.5) * dy

        for step in range(max_steps):
            current_vol = model.get_volume_fraction()
            if current_vol <= target_vol + volume_tolerance:
                break

            density = model.to_density_grid()
            fea_res = self.solver.solve(density, penalty=1.0)
            strain_energy = fea_res.element_strain_energies

            # Only consider solid material currently present
            valid_mask = density > 0.5
            if not np.any(valid_mask):
                break

            # Find elements with lowest strain energy among valid solid elements
            masked_energy = np.where(valid_mask, strain_energy, np.inf)

            # Sort candidate points
            flat_indices = np.argsort(masked_energy.flatten())

            placed = False
            for idx in flat_indices[:15]:
                j = idx // self.problem.nx
                i = idx % self.problem.nx
                cand_x = float(xs[i])
                cand_y = float(ys[j])

                # Don't place cuts right on top of supports or loads
                near_support = any(
                    np.hypot(cand_x - s.x, cand_y - s.y) < cut_radius * 1.2
                    for s in self.problem.supports
                )
                near_load = any(
                    np.hypot(cand_x - l.x, cand_y - l.y) < cut_radius * 1.2
                    for l in self.problem.loads
                )
                if near_support or near_load:
                    continue

                # Dynamically size cut to avoid overshooting volume target
                remaining_vol = current_vol - target_vol
                domain_area = self.problem.width * self.problem.height
                max_cut_area = remaining_vol * domain_area
                r = min(cut_radius, float(np.sqrt(max_cut_area / np.pi) * 1.1))
                r = max(1.5, r)

                success = model.cut_circle(cand_x, cand_y, r)
                if success:
                    # Verify cut does not sever primary load path
                    test_grid = model.to_density_grid()
                    test_fea = self.solver.solve(test_grid, penalty=1.0)
                    if test_fea.compliance > fea_res.compliance * 8.0:
                        model.undo()  # Severed structure; revert
                        continue
                    placed = True
                    break

            if not placed:
                break

        return model
