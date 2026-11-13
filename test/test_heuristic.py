from src.agents.heuristic_agent import HeuristicCADDesigner
from src.agents.problem import make_cantilever_beam


def test_heuristic_cad_designer():
    problem = make_cantilever_beam(
        width=40.0, height=20.0, nx=20, ny=10, target_volume_fraction=0.60
    )
    designer = HeuristicCADDesigner(problem)
    model = designer.design(max_steps=8, cut_radius=3.5)

    assert len(model.history) > 0
    # Volume fraction should have been reduced towards target
    assert model.get_volume_fraction() < 0.95
