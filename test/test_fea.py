import numpy as np
from src.agents.fea import FEASolver
from src.agents.problem import make_cantilever_beam


def test_fea_cantilever_solve():
    problem = make_cantilever_beam(width=60.0, height=30.0, nx=30, ny=15)
    solver = FEASolver(problem)

    # Solid block test
    density = np.ones((problem.ny, problem.nx), dtype=float)
    result = solver.solve(density)

    assert result.compliance > 0.0
    assert not np.isnan(result.compliance)
    assert result.displacements.shape == (2 * (solver.nx + 1) * (solver.ny + 1),)
    assert result.von_mises_stress.shape == (problem.ny, problem.nx)

    # Tip displacement should be negative (downward load)
    # The load was applied at (width, 0.0) with fy = -1.0
    # Node at (width, 0) is j=0, i=nx -> node_idx = 0*(nx+1) + nx = nx
    # DOF for y is 2 * nx + 1
    tip_y_dof = 2 * solver.nx + 1
    assert result.displacements[tip_y_dof] < 0.0
