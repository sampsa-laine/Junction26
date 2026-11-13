from src.agents.problem import make_cantilever_beam
from src.agents.simp import SIMPOptimizer


def test_simp_optimization_convergence():
    problem = make_cantilever_beam(
        width=30.0, height=15.0, nx=30, ny=15, target_volume_fraction=0.50
    )
    opt = SIMPOptimizer(problem)
    result = opt.optimize(max_iterations=15)

    assert result.iterations > 0
    assert result.minimum_compliance > 0.0
    assert len(result.compliance_history) == result.iterations
    # Check that volume fraction is approximately met
    assert abs(result.volume_fraction - 0.50) < 0.05
    # Compliance should generally decrease or stabilize
    assert result.compliance_history[-1] <= result.compliance_history[0] * 1.5
