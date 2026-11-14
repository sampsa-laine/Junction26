from src.agents.csg import CADModel
from src.agents.problem import make_cantilever_beam, make_suspension_arm


def test_cad_model_operations():
    problem = make_cantilever_beam(width=60.0, height=30.0, nx=20, ny=10)
    model = CADModel(problem)

    assert abs(model.get_volume_fraction() - 1.0) < 1e-4

    # Cut a circle
    ok = model.cut_circle(x=30.0, y=15.0, radius=5.0)
    assert ok
    assert model.get_volume_fraction() < 1.0
    assert len(model.history) == 1

    # Rasterize
    grid = model.to_density_grid()
    assert grid.shape == (problem.ny, problem.nx)
    assert 0.0 <= grid.min() and grid.max() <= 1.0
    # Center element should have reduced density
    center_j = problem.ny // 2
    center_i = problem.nx // 2
    assert grid[center_j, center_i] < 0.5

    # Test undo
    model.undo()
    assert len(model.history) == 0
    assert abs(model.get_volume_fraction() - 1.0) < 1e-4


def test_cad_keepouts():
    problem = make_suspension_arm(width=80.0, height=40.0)
    model = CADModel(problem)

    # Trying to cut a circle exactly over the solid pivot hole (10, 20, r=5)
    # The cut should either be rejected or clipped around the keepout
    init_vol = model.get_volume_fraction()
    model.cut_circle(x=10.0, y=20.0, radius=4.0)
    # The keepout must remain intact
    assert not model.solid_keepouts.is_empty
