from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problem import DesignProblem


@dataclass
class FEAResult:
    displacements: np.ndarray  # Shape: (2 * num_nodes,)
    compliance: float
    element_strain_energies: np.ndarray  # Shape: (ny, nx)
    von_mises_stress: np.ndarray  # Shape: (ny, nx)


def compute_q4_element_stiffness(
    dx: float, dy: float, youngs_modulus: float = 1.0, poissons_ratio: float = 0.3
) -> np.ndarray:
    """Computes exact 8x8 stiffness matrix for a 4-node rectangular element using 2x2 Gauss quadrature."""
    e = youngs_modulus
    nu = poissons_ratio

    # Plane stress constitutive matrix
    factor = e / (1.0 - nu**2)
    d_mat = factor * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )

    a = dx / 2.0
    b = dy / 2.0
    det_j = a * b

    # Gauss quadrature points
    g_pt = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-g_pt, -g_pt), (g_pt, -g_pt), (g_pt, g_pt), (-g_pt, g_pt)]

    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    k_elem = np.zeros((8, 8), dtype=float)

    for xi, eta in gauss_pts:
        # Derivatives of shape functions w.r.t natural coords
        d_n = np.zeros((4, 2), dtype=float)
        for i in range(4):
            xi_i, eta_i = node_coords[i]
            d_n[i, 0] = 0.25 * xi_i * (1.0 + eta_i * eta)
            d_n[i, 1] = 0.25 * eta_i * (1.0 + xi_i * xi)

        # Derivatives w.r.t physical coords
        d_n_dx = d_n[:, 0] / a
        d_n_dy = d_n[:, 1] / b

        # Strain-displacement matrix B (3 x 8)
        b_mat = np.zeros((3, 8), dtype=float)
        for i in range(4):
            b_mat[0, 2 * i] = d_n_dx[i]
            b_mat[1, 2 * i + 1] = d_n_dy[i]
            b_mat[2, 2 * i] = d_n_dy[i]
            b_mat[2, 2 * i + 1] = d_n_dx[i]

        k_elem += b_mat.T @ d_mat @ b_mat * det_j

    return k_elem


class FEASolver:
    """Vectorized 2D Quad-4 linear elastic finite element solver."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.nx = problem.nx
        self.ny = problem.ny
        self.dx = problem.width / problem.nx
        self.dy = problem.height / problem.ny
        self.num_nodes = (self.nx + 1) * (self.ny + 1)
        self.num_dofs = 2 * self.num_nodes

        self.k_elem = compute_q4_element_stiffness(
            self.dx, self.dy, problem.youngs_modulus, problem.poissons_ratio
        )

        self._build_mesh_connectivity()
        self._build_boundary_conditions()

    def _build_mesh_connectivity(self) -> None:
        """Builds DOF mappings for vectorized global matrix assembly."""
        # Node index grid: (ny+1) x (nx+1)
        # Node ordering: n(j, i) = j * (nx + 1) + i where j in [0, ny], i in [0, nx]
        node_indices = np.arange(self.num_nodes).reshape((self.ny + 1, self.nx + 1))

        # Element node indices: [bottom-left, bottom-right, top-right, top-left]
        n1 = node_indices[:-1, :-1].flatten()
        n2 = node_indices[:-1, 1:].flatten()
        n3 = node_indices[1:, 1:].flatten()
        n4 = node_indices[1:, :-1].flatten()

        # 8 DOFs per element
        edof = np.zeros((self.ny * self.nx, 8), dtype=int)
        edof[:, 0] = 2 * n1
        edof[:, 1] = 2 * n1 + 1
        edof[:, 2] = 2 * n2
        edof[:, 3] = 2 * n2 + 1
        edof[:, 4] = 2 * n3
        edof[:, 5] = 2 * n3 + 1
        edof[:, 6] = 2 * n4
        edof[:, 7] = 2 * n4 + 1
        self.edof = edof

        # Sparse assembly index patterns
        self.i_k = np.kron(edof, np.ones((8, 1), dtype=int)).flatten()
        self.j_k = np.kron(edof, np.ones((1, 8), dtype=int)).flatten()

    def _build_boundary_conditions(self) -> None:
        """Locates fixed DOFs and applies external forces based on problem definition."""
        fixed_dofs: list[int] = []
        force_vector = np.zeros(self.num_dofs, dtype=float)

        # Node coordinates
        xs = np.linspace(0.0, self.problem.width, self.nx + 1)
        ys = np.linspace(0.0, self.problem.height, self.ny + 1)

        # Supports
        for supp in self.problem.supports:
            for j, y_coord in enumerate(ys):
                for i, x_coord in enumerate(xs):
                    dist = np.hypot(x_coord - supp.x, y_coord - supp.y)
                    if dist <= supp.tolerance:
                        node_idx = j * (self.nx + 1) + i
                        if supp.fixed_x:
                            fixed_dofs.append(2 * node_idx)
                        if supp.fixed_y:
                            fixed_dofs.append(2 * node_idx + 1)

        self.fixed_dofs = np.unique(np.array(fixed_dofs, dtype=int))
        all_dofs = np.arange(self.num_dofs)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)

        # Loads
        for load in self.problem.loads:
            # Find closest node
            i = int(np.clip(round(load.x / self.dx), 0, self.nx))
            j = int(np.clip(round(load.y / self.dy), 0, self.ny))
            node_idx = j * (self.nx + 1) + i
            force_vector[2 * node_idx] += load.fx
            force_vector[2 * node_idx + 1] += load.fy

        self.force_vector = force_vector

    def solve(
        self, density_field: np.ndarray, emin: float = 1e-6, penalty: float = 3.0
    ) -> FEAResult:
        """Solves the FEA system for a given element density field."""
        rho = np.clip(density_field, 0.0, 1.0).flatten()

        # SIMP stiffness scaling: E = emin + rho^p * (1 - emin)
        e_factors = emin + (rho**penalty) * (1.0 - emin)

        # Assemble global stiffness matrix
        s_k = (
            self.k_elem.flatten()[np.newaxis, :] * e_factors[:, np.newaxis]
        ).flatten()
        k_global = sp.coo_matrix(
            (s_k, (self.i_k, self.j_k)), shape=(self.num_dofs, self.num_dofs)
        ).tocsr()

        # Partition matrix for free DOFs
        k_free = k_global[self.free_dofs, :][:, self.free_dofs]
        f_free = self.force_vector[self.free_dofs]

        # Solve system
        u_free = spla.spsolve(k_free, f_free)

        # Reconstruct full displacement vector
        u_full = np.zeros(self.num_dofs, dtype=float)
        u_full[self.free_dofs] = u_free

        # Calculate element strain energies and compliance
        u_elem = u_full[self.edof]  # (num_elem, 8)
        # Element compliance: c_e = u_e^T * (E_e * k0) * u_e
        strain_energy = np.sum((u_elem @ self.k_elem) * u_elem, axis=1)
        elem_compliance = e_factors * strain_energy
        total_compliance = float(np.sum(elem_compliance))

        # Von Mises stress estimation from element strain energy
        # For plane stress, strain energy density U_d approx proportional to von Mises stress squared
        elem_area = self.dx * self.dy
        energy_density = strain_energy / elem_area
        approx_von_mises = np.sqrt(2.0 * np.maximum(0.0, energy_density)) * rho

        return FEAResult(
            displacements=u_full,
            compliance=total_compliance,
            element_strain_energies=strain_energy.reshape((self.ny, self.nx)),
            von_mises_stress=approx_von_mises.reshape((self.ny, self.nx)),
        )
