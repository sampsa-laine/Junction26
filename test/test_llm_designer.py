from src.agents.llm_designer import LLMCADDesigner
from src.agents.problem import make_cantilever_beam


def test_llm_cad_designer():
    problem = make_cantilever_beam(
        width=40.0, height=20.0, nx=20, ny=10, target_volume_fraction=0.60
    )
    designer = LLMCADDesigner(problem)
    result = designer.design(max_steps=5)

    assert len(result.transcript) > 0
    assert result.cad_model.get_volume_fraction() < 1.0
    for record in result.transcript:
        assert record.compliance > 0.0
        assert len(record.thought) > 0
