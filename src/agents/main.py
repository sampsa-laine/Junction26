import os
import subprocess
import sys

from src.agents.csg import CADModel
from src.agents.fea import FEASolver
from src.agents.heuristic_agent import HeuristicCADDesigner
from src.agents.llm_designer import LLMCADDesigner
from src.agents.metrics import evaluate_cad_model, evaluate_design
from src.agents.problem import make_cantilever_beam
from src.agents.simp import SIMPOptimizer


def main():
    """CLI entrypoint for 2D CAD TopOpt Arena."""
    print("=" * 65)
    print("  🏗️  2D CAD TOPOLOGY OPTIMIZATION ARENA (JUNCTION 2026)")
    print("=" * 65)

    problem = make_cantilever_beam(
        width=60.0, height=30.0, nx=40, ny=20, target_volume_fraction=0.40
    )
    print(
        f"Problem: {problem.name} ({problem.width:.0f}x{problem.height:.0f}) | Target Mass: {problem.target_volume_fraction * 100:.0f}%"
    )
    print("-" * 65)

    solver = FEASolver(problem)

    print("1. Computing Classical SIMP Topology Optimization...")
    simp_opt = SIMPOptimizer(problem)
    simp_res = simp_opt.optimize(max_iterations=20)
    score_simp = evaluate_design(
        "Classical SIMP (Optimum)",
        simp_res.optimal_density,
        problem,
        simp_res,
        solver,
    )

    print("2. Running LLM CAD Designer Agent...")
    llm_designer = LLMCADDesigner(problem)
    llm_res = llm_designer.design(max_steps=10)
    score_llm = evaluate_cad_model(
        "LLM CAD Agent", llm_res.cad_model, problem, simp_res, solver
    )

    print("3. Running Dedicated Heuristic Model...")
    heur_designer = HeuristicCADDesigner(problem)
    heur_model = heur_designer.design(max_steps=12)
    score_heur = evaluate_cad_model(
        "Dedicated Model", heur_model, problem, simp_res, solver
    )

    # Empty baseline model
    base_model = CADModel(problem)
    score_base = evaluate_cad_model(
        "Solid Stock Plate", base_model, problem, simp_res, solver
    )

    print("\n" + "=" * 65)
    print(
        f"{'Competitor':<26} | {'Comp (Stiff)':<12} | {'Gap %':<7} | {'Vol %':<7} | {'Score'}"
    )
    print("-" * 65)
    scores = [score_simp, score_llm, score_heur, score_base]
    for s in scores:
        gap_str = (
            "0.0%"
            if s.designer_name.startswith("Classical")
            else f"+{s.optimality_gap_percent:.1f}%"
        )
        print(
            f"{s.designer_name:<26} | {s.compliance:<12.2f} | {gap_str:<7} | {s.volume_fraction * 100:>5.1f}% | {s.overall_score:>5.1f}/100"
        )
    print("=" * 65)
    print(
        "\nTo launch the interactive UI dashboard, run:\n  uv run arena\nor\n  uv run streamlit run src/agents/app.py\n"
    )


def run_app():
    """Launches the Streamlit web dashboard."""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run(["streamlit", "run", app_path] + sys.argv[1:])


if __name__ == "__main__":
    main()
