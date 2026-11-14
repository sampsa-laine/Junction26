from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problem import DesignProblem


@dataclass
class FEAResult:
    displacements: np.ndarray  # Primary load case displacements: (2 * num_nodes,)
    compliance: float
    element_strain_energies: np.ndarray  # Summed over load cases: (ny, nx)
    von_mises_stress: np.ndarray  # Envelope max over load cases: (ny, nx)
    case_compliances: list[float] = field(default_factory=list)
    case_max_stresses: list[float] = field(default_factory=list)


def compute_q4_element_matrices(
    dx: float, dy: float, youngs_modulus: float = 1.0, poissons_ratio: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Computes exact 8x8 Q4 stiffness matrix and (4, 3, 8) Gauss-point stress matrices D*B."""
    e = youngs_modulus
    nu = poissons_ratio

    factor = e / (1.0 - nu**2)
    d_mat = factor * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )

    a = dx / 2.0
    b = dy / 2.0
    det_j = a * b

    g_pt = 1.0 / np.sqrt(3.0)
    gauss_pts = [(-g_pt, -g_pt), (g_pt, -g_pt), (g_pt, g_pt), (-g_pt, g_pt)]
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    k_elem = np.zeros((8, 8), dtype=float)
    db_gauss = np.zeros((4, 3, 8), dtype=float)

    for g_idx, (xi, eta) in enumerate(gauss_pts):
        d_n = np.zeros((4, 2), dtype=float)
        for i in range(4):
            xi_i, eta_i = node_coords[i]
            d_n[i, 0] = 0.25 * xi_i * (1.0 + eta_i * eta)
            d_n[i, 1] = 0.25 * eta_i * (1.0 + xi_i * xi)

        d_n_dx = d_n[:, 0] / a
        d_n_dy = d_n[:, 1] / b

        b_mat = np.zeros((3, 8), dtype=float)
        for i in range(4):
            b_mat[0, 2 * i] = d_n_dx[i]
            b_mat[1, 2 * i + 1] = d_n_dy[i]
            b_mat[2, 2 * i] = d_n_dy[i]
            b_mat[2, 2 * i + 1] = d_n_dx[i]

        k_elem += b_mat.T @ d_mat @ b_mat * det_j
        db_gauss[g_idx] = d_mat @ b_mat

    return k_elem, db_gauss


def compute_q4_element_stiffness(
    dx: float, dy: float, youngs_modulus: float = 1.0, poissons_ratio: float = 0.3
) -> np.ndarray:
    """Computes exact 8x8 stiffness matrix for a 4-node rectangular element."""
    k_elem, _ = compute_q4_element_matrices(dx, dy, youngs_modulus, poissons_ratio)
    return k_elem


class FEASolver:
    """Vectorized 2D Quad-4 linear elastic finite element solver supporting multi-load cases."""

    def __init__(self, problem: DesignProblem):
        self.problem = problem
        self.nx = problem.nx
        self.ny = problem.ny
        self.dx = problem.width / problem.nx
        self.dy = problem.height / problem.ny
        self.num_nodes = (self.nx + 1) * (self.ny + 1)
        self.num_dofs = 2 * self.num_nodes

        self.k_elem, self.db_gauss = compute_q4_element_matrices(
            self.dx, self.dy, problem.youngs_modulus, problem.poissons_ratio
        )

        self._build_mesh_connectivity()
        self._build_boundary_conditions()

    def _build_mesh_connectivity(self) -> None:
        node_indices = np.arange(self.num_nodes).reshape((self.ny + 1, self.nx + 1))

        n1 = node_indices[:-1, :-1].flatten()
        n2 = node_indices[:-1, 1:].flatten()
        n3 = node_indices[1:, 1:].flatten()
        n4 = node_indices[1:, :-1].flatten()

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

        self.i_k = np.kron(edof, np.ones((8, 1), dtype=int)).flatten()
        self.j_k = np.kron(edof, np.ones((1, 8), dtype=int)).flatten()

    def _build_boundary_conditions(self) -> None:
        fixed_dofs: list[int] = []

        xs = np.linspace(0.0, self.problem.width, self.nx + 1)
        ys = np.linspace(0.0, self.problem.height, self.ny + 1)
        x_grid, y_grid = np.meshgrid(xs, ys)

        for supp in self.problem.supports:
            dists = np.hypot(x_grid - supp.x, y_grid - supp.y)
            matched_j, matched_i = np.where(dists <= supp.tolerance)
            node_ids = matched_j * (self.nx + 1) + matched_i
            if supp.fixed_x:
                fixed_dofs.extend((2 * node_ids).tolist())
            if supp.fixed_y:
                fixed_dofs.extend((2 * node_ids + 1).tolist())

        self.fixed_dofs = np.unique(np.array(fixed_dofs, dtype=int))
        all_dofs = np.arange(self.num_dofs)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)

        cases = self.problem.effective_load_cases
        num_cases = max(1, len(cases))
        force_matrix = np.zeros((self.num_dofs, num_cases), dtype=float)

        for c_idx, load_list in enumerate(cases):
            for load in load_list:
                i = int(np.clip(round(load.x / self.dx), 0, self.nx))
                j = int(np.clip(round(load.y / self.dy), 0, self.ny))
                node_idx = j * (self.nx + 1) + i
                force_matrix[2 * node_idx, c_idx] += load.fx
                force_matrix[2 * node_idx + 1, c_idx] += load.fy

        self.force_matrix = force_matrix
        self.force_vector = force_matrix[:, 0]

    def solve(
        self, density_field: np.ndarray, emin: float = 1e-6, penalty: float = 3.0
    ) -> FEAResult:
        """Solves the linear elastic FEA system across all load cases simultaneously."""
        rho = np.clip(density_field, 0.0, 1.0).flatten()
        e_factors = emin + (rho**penalty) * (1.0 - emin)

        s_k = (
            self.k_elem.flatten()[np.newaxis, :] * e_factors[:, np.newaxis]
        ).flatten()
        k_global = sp.coo_matrix(
            (s_k, (self.i_k, self.j_k)), shape=(self.num_dofs, self.num_dofs)
        ).tocsr()

        k_free = k_global[self.free_dofs, :][:, self.free_dofs]
        f_free = self.force_matrix[self.free_dofs, :]
        num_cases = f_free.shape[1]

        u_free_raw = spla.spsolve(k_free, f_free)
        u_free = np.asarray(u_free_raw, dtype=float).reshape(
            (len(self.free_dofs), num_cases)
        )

        u_full = np.zeros((self.num_dofs, num_cases), dtype=float)
        u_full[self.free_dofs, :] = u_free

        num_elems = self.ny * self.nx
        total_strain_energy = np.zeros(num_elems, dtype=float)
        max_vm_envelope = np.zeros(num_elems, dtype=float)
        case_compliances: list[float] = []
        case_max_stresses: list[float] = []

        # Stress interpolation scaling (penalized E/E0 = rho for physical solid elements)
        stress_scale = np.where(rho > 1e-3, rho**0.5, 0.0)

        for c_idx in range(num_cases):
            u_elem = u_full[self.edof, c_idx]  # (num_elems, 8)
            se_case = np.sum((u_elem @ self.k_elem) * u_elem, axis=1)
            comp_case = float(np.sum(e_factors * se_case))
            case_compliances.append(comp_case)
            total_strain_energy += se_case

            # Evaluate plane-stress von Mises across the 4 Gauss points: (4, num_elems, 3)
            sigmas = np.einsum("gij,ej->gei", self.db_gauss, u_elem)
            s_xx = sigmas[:, :, 0]
            s_yy = sigmas[:, :, 1]
            s_xy = sigmas[:, :, 2]
            vm_gauss = np.sqrt(
                np.maximum(0.0, s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)
            )
            vm_elem = np.max(vm_gauss, axis=0) * stress_scale
            case_max_stresses.append(float(np.max(vm_elem)))
            max_vm_envelope = np.maximum(max_vm_envelope, vm_elem)

        return FEAResult(
            displacements=u_full[:, 0],
            compliance=float(sum(case_compliances)),
            element_strain_energies=total_strain_energy.reshape((self.ny, self.nx)),
            von_mises_stress=max_vm_envelope.reshape((self.ny, self.nx)),
            case_compliances=case_compliances,
            case_max_stresses=case_max_stresses,
        )
