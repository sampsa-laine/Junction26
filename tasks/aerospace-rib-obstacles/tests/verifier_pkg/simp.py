from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from .fea import FEASolver
from .problem import DesignProblem, KeepOutType


@dataclass
class SIMPResult:
    optimal_density: np.ndarray  # Shape: (ny, nx)
    minimum_compliance: float
    iterations: int
    compliance_history: list[float]
    volume_fraction: float


class SIMPOptimizer:
    """Generalized Sigmund 88-line SIMP topology optimization solver."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.nx = problem.nx
        self.ny = problem.ny
        self.solver = FEASolver(problem)
        self.dx = problem.width / problem.nx
        self.dy = problem.height / problem.ny

        self._build_filter(problem.filter_radius)
        self._build_keepout_masks()

    def _build_filter(self, rmin: float) -> None:
        """Precomputes mesh-independent spatial sensitivity filter."""
        num_elems = self.ny * self.nx
        i_h: list[int] = []
        j_h: list[int] = []
        s_h: list[float] = []

        # Element centers
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dy
        x_grid, y_grid = np.meshgrid(xs, ys)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        for e in range(num_elems):
            # Compute distance to all other elements
            dists = np.hypot(x_flat - x_flat[e], y_flat - y_flat[e])
            neighbors = np.where(dists < rmin)[0]
            weights = rmin - dists[neighbors]

            for n_idx, w in zip(neighbors, weights):
                i_h.append(int(e))
                j_h.append(int(n_idx))
                s_h.append(float(w))

        self.h_mat = sp.coo_matrix(
            (s_h, (i_h, j_h)), shape=(num_elems, num_elems)
        ).tocsr()
        self.hs = np.array(self.h_mat.sum(axis=1)).flatten()

    def _build_keepout_masks(self) -> None:
        """Identifies fixed solid and void elements from problem keepouts."""
        solid_mask = np.zeros((self.ny, self.nx), dtype=bool)
        void_mask = np.zeros((self.ny, self.nx), dtype=bool)

        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dy

        for keepout in self.problem.keepouts:
            if keepout.shape == "circle":
                cx, cy, r = keepout.params
                for j, y in enumerate(ys):
                    for i, x in enumerate(xs):
                        if np.hypot(x - cx, y - cy) <= r:
                            if keepout.keepout_type == KeepOutType.SOLID:
                                solid_mask[j, i] = True
                            else:
                                void_mask[j, i] = True
            elif keepout.shape == "rect":
                rx, ry, rw, rh = keepout.params
                for j, y in enumerate(ys):
                    for i, x in enumerate(xs):
                        if rx <= x <= rx + rw and ry <= y <= ry + rh:
                            if keepout.keepout_type == KeepOutType.SOLID:
                                solid_mask[j, i] = True
                            else:
                                void_mask[j, i] = True

        self.solid_mask = solid_mask.flatten()
        self.void_mask = void_mask.flatten()
        self.designable_mask = ~(self.solid_mask | self.void_mask)

    def optimize(
        self,
        max_iterations: int = 50,
        change_tol: float = 0.01,
        move: float = 0.2,
    ) -> SIMPResult:
        """Executes Optimality Criteria (OC) SIMP optimization loop."""
        num_elems = self.ny * self.nx
        volfrac = self.problem.target_volume_fraction
        p = self.problem.penalty

        # Initial uniform density
        x = np.full(num_elems, volfrac, dtype=float)
        x[self.solid_mask] = 1.0
        x[self.void_mask] = 0.0

        compliance_history: list[float] = []
        it = 0

        for it in range(max_iterations):
            # FEA solve
            res = self.solver.solve(x.reshape((self.ny, self.nx)), penalty=p)
            compliance_history.append(res.compliance)

            # Strain energy sensitivities
            # dc = -p * x^(p-1) * u_e^T k0 u_e
            se = res.element_strain_energies.flatten()
            dc = -p * (x ** (p - 1.0)) * se

            # Sensitivity filtering: dc_filtered = (H * (x * dc)) / (x * Hs)
            dc_filtered = np.array(self.h_mat.dot(x * dc)).flatten() / (
                np.maximum(1e-12, x) * self.hs
            )

            # Optimality Criteria (OC) Bisection Search for Lagrange Multiplier
            l1 = 0.0
            l2 = 1e9
            x_candidate = np.copy(x)

            # Target volume on designable domain
            fixed_volume = np.sum(x[self.solid_mask])
            target_design_vol = volfrac * num_elems - fixed_volume

            while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
                l_mid = 0.5 * (l2 + l1)
                # Bisection update
                step = x * np.sqrt(np.maximum(0.0, -dc_filtered / l_mid))
                x_candidate = np.maximum(
                    0.0,
                    np.maximum(
                        x - move,
                        np.minimum(1.0, np.minimum(x + move, step)),
                    ),
                )
                x_candidate[self.solid_mask] = 1.0
                x_candidate[self.void_mask] = 0.0

                current_design_vol = np.sum(x_candidate[self.designable_mask])
                if current_design_vol > target_design_vol:
                    l1 = l_mid
                else:
                    l2 = l_mid

            change = float(np.max(np.abs(x_candidate - x)))
            x = x_candidate

            if change < change_tol and it > 5:
                break

        final_res = self.solver.solve(x.reshape((self.ny, self.nx)), penalty=1.0)
        actual_vol = float(np.mean(x))

        return SIMPResult(
            optimal_density=x.reshape((self.ny, self.nx)),
            minimum_compliance=final_res.compliance,
            iterations=it + 1,
            compliance_history=compliance_history,
            volume_fraction=actual_vol,
        )
