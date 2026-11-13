import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.agents.csg import CADModel
from src.agents.export import export_cad_to_dxf, export_cad_to_svg
from src.agents.fea import FEASolver
from src.agents.heuristic_agent import HeuristicCADDesigner
from src.agents.llm_designer import LLMCADDesigner
from src.agents.metrics import evaluate_cad_model, evaluate_design
from src.agents.problem import (
    make_cantilever_beam,
    make_l_bracket,
    make_mbb_beam,
    make_suspension_arm,
)
from src.agents.simp import SIMPOptimizer


def render_component_plot(
    density: np.ndarray,
    title: str,
    subtitle: str = "",
    stress: np.ndarray | None = None,
):
    """Renders a 2D density/stress heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    if stress is not None:
        im = ax.imshow(
            stress,
            origin="lower",
            cmap="turbo",
            interpolation="nearest",
            alpha=0.9,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Stress")
    else:
        ax.imshow(
            1.0 - density,
            origin="lower",
            cmap="gray",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )

    ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="2D CAD TopOpt Arena", page_icon="🏗️", layout="wide")

    st.title("🏗️ 2D CAD Topology Optimization Arena")
    st.caption(
        "Benchmarking AI Agents, Classical TopOpt, and Human Designers on Structural CAD Components"
    )

    # Sidebar: Problem Selection
    st.sidebar.header("Design Problem")
    preset = st.sidebar.selectbox(
        "Preset Benchmark",
        ["Cantilever Beam", "MBB Beam", "L-Bracket", "Suspension Arm"],
    )
    vol_target = st.sidebar.slider("Target Volume Fraction", 0.20, 0.70, 0.40, 0.05)

    if preset == "Cantilever Beam":
        problem = make_cantilever_beam(
            width=60.0, height=30.0, nx=40, ny=20, target_volume_fraction=vol_target
        )
    elif preset == "MBB Beam":
        problem = make_mbb_beam(
            width=60.0, height=20.0, nx=40, ny=15, target_volume_fraction=vol_target
        )
    elif preset == "L-Bracket":
        problem = make_l_bracket(
            width=40.0, height=40.0, nx=30, ny=30, target_volume_fraction=vol_target
        )
    else:
        problem = make_suspension_arm(
            width=60.0, height=30.0, nx=40, ny=20, target_volume_fraction=vol_target
        )

    solver = FEASolver(problem)

    # Human Designer State in Session State
    if (
        "human_model" not in st.session_state
        or st.session_state.get("current_preset") != preset
    ):
        st.session_state["human_model"] = CADModel(problem)
        st.session_state["current_preset"] = preset

    # Tabs
    tab_arena, tab_human, tab_llm_log, tab_export = st.tabs(
        ["🏆 4-Way Arena", "✏️ Human Canvas", "🧠 LLM Reasoning", "💾 CAD Export"]
    )

    with tab_arena:
        if st.button("🚀 Run Full 4-Way Benchmark", type="primary"):
            with st.spinner("Computing Classical SIMP TopOpt theoretical optimum..."):
                simp_opt = SIMPOptimizer(problem)
                simp_res = simp_opt.optimize(max_iterations=20)
                st.session_state["simp_res"] = simp_res

            with st.spinner("Running LLM CAD Designer agent..."):
                llm_designer = LLMCADDesigner(problem)
                llm_res = llm_designer.design(max_steps=6)
                st.session_state["llm_res"] = llm_res

            with st.spinner("Running Dedicated Model (Heuristic)..."):
                heur_designer = HeuristicCADDesigner(problem)
                heur_model = heur_designer.design(max_steps=8)
                st.session_state["heur_model"] = heur_model

        if "simp_res" in st.session_state:
            simp_res = st.session_state["simp_res"]
            llm_res = st.session_state["llm_res"]
            heur_model = st.session_state["heur_model"]
            human_model = st.session_state["human_model"]

            # Evaluations
            score_simp = evaluate_design(
                "1. Classical SIMP (Theoretical)",
                simp_res.optimal_density,
                problem,
                simp_res,
                solver,
            )
            score_llm = evaluate_cad_model(
                "2. LLM CAD Designer", llm_res.cad_model, problem, simp_res, solver
            )
            score_heur = evaluate_cad_model(
                "3. Dedicated Model", heur_model, problem, simp_res, solver
            )
            score_human = evaluate_cad_model(
                "4. Human Designer", human_model, problem, simp_res, solver
            )

            # Display 4 Columns
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.subheader("1. Classical SIMP")
                st.caption("Theoretical Upper Bound (Non-CAD)")
                fig = render_component_plot(
                    simp_res.optimal_density,
                    "Optimal Topology",
                    f"C* = {simp_res.minimum_compliance:.2f} | Vol = {score_simp.volume_fraction:.2f}",
                )
                st.pyplot(fig)
                st.metric("Compliance", f"{score_simp.compliance:.2f}")
                st.metric("Optimality Gap", "0.0% (Optimum)")
                st.metric("Score", f"{score_simp.overall_score:.1f}/100")

            with c2:
                st.subheader("2. LLM CAD Agent")
                st.caption("Iterative ReAct Tool Use")
                fea_llm = solver.solve(llm_res.cad_model.to_density_grid())
                fig = render_component_plot(
                    llm_res.cad_model.to_density_grid(),
                    "LLM CAD Geometry",
                    f"C = {score_llm.compliance:.2f} | Vol = {score_llm.volume_fraction:.2f}",
                    stress=fea_llm.von_mises_stress,
                )
                st.pyplot(fig)
                st.metric("Compliance", f"{score_llm.compliance:.2f}")
                st.metric("Optimality Gap", f"+{score_llm.optimality_gap_percent:.1f}%")
                st.metric("Score", f"{score_llm.overall_score:.1f}/100")

            with c3:
                st.subheader("3. Dedicated Model")
                st.caption("Stress-Guided Heuristic")
                fea_heur = solver.solve(heur_model.to_density_grid())
                fig = render_component_plot(
                    heur_model.to_density_grid(),
                    "Heuristic CAD Geometry",
                    f"C = {score_heur.compliance:.2f} | Vol = {score_heur.volume_fraction:.2f}",
                    stress=fea_heur.von_mises_stress,
                )
                st.pyplot(fig)
                st.metric("Compliance", f"{score_heur.compliance:.2f}")
                st.metric(
                    "Optimality Gap", f"+{score_heur.optimality_gap_percent:.1f}%"
                )
                st.metric("Score", f"{score_heur.overall_score:.1f}/100")

            with c4:
                st.subheader("4. Human Designer")
                st.caption("Interactive CAD Canvas")
                fea_human = solver.solve(human_model.to_density_grid())
                fig = render_component_plot(
                    human_model.to_density_grid(),
                    "Human CAD Geometry",
                    f"C = {score_human.compliance:.2f} | Vol = {score_human.volume_fraction:.2f}",
                    stress=fea_human.von_mises_stress,
                )
                st.pyplot(fig)
                st.metric("Compliance", f"{score_human.compliance:.2f}")
                st.metric(
                    "Optimality Gap", f"+{score_human.optimality_gap_percent:.1f}%"
                )
                st.metric("Score", f"{score_human.overall_score:.1f}/100")

            # Scoreboard Table
            st.divider()
            st.subheader("📊 Arena Scoreboard")
            scoreboard_data = [
                {
                    "Rank": 1,
                    "Competitor": "SIMP TopOpt (Theoretical Limit)",
                    "Type": "Classical OC",
                    "Compliance": f"{score_simp.compliance:.2f}",
                    "Optimality Gap": "0.0%",
                    "Volume Fraction": f"{score_simp.volume_fraction:.2f}",
                    "Max Stress": f"{score_simp.max_von_mises_stress:.2f}",
                    "Overall Score": f"{score_simp.overall_score:.1f}",
                },
                {
                    "Rank": 2,
                    "Competitor": "LLM CAD Designer Agent",
                    "Type": "ReAct AI",
                    "Compliance": f"{score_llm.compliance:.2f}",
                    "Optimality Gap": f"+{score_llm.optimality_gap_percent:.1f}%",
                    "Volume Fraction": f"{score_llm.volume_fraction:.2f}",
                    "Max Stress": f"{score_llm.max_von_mises_stress:.2f}",
                    "Overall Score": f"{score_llm.overall_score:.1f}",
                },
                {
                    "Rank": 3,
                    "Competitor": "Dedicated Model",
                    "Type": "Strain Heuristic",
                    "Compliance": f"{score_heur.compliance:.2f}",
                    "Optimality Gap": f"+{score_heur.optimality_gap_percent:.1f}%",
                    "Volume Fraction": f"{score_heur.volume_fraction:.2f}",
                    "Max Stress": f"{score_heur.max_von_mises_stress:.2f}",
                    "Overall Score": f"{score_heur.overall_score:.1f}",
                },
                {
                    "Rank": 4,
                    "Competitor": "Human Designer",
                    "Type": "Manual Cuts",
                    "Compliance": f"{score_human.compliance:.2f}",
                    "Optimality Gap": f"+{score_human.optimality_gap_percent:.1f}%",
                    "Volume Fraction": f"{score_human.volume_fraction:.2f}",
                    "Max Stress": f"{score_human.max_von_mises_stress:.2f}",
                    "Overall Score": f"{score_human.overall_score:.1f}",
                },
            ]
            st.table(scoreboard_data)
        else:
            st.info(
                "Click **'🚀 Run Full 4-Way Benchmark'** above to compute the arena comparisons!"
            )

    with tab_human:
        st.subheader("✏️ Human Designer Canvas")
        st.write(
            "Place holes or slots manually to see if you can beat the AI or match SIMP TopOpt!"
        )
        human_model: CADModel = st.session_state["human_model"]

        col_tools, col_view = st.columns([1, 2])
        with col_tools:
            cut_type = st.radio("Tool", ["Circular Hole", "Rectangular Cutout"])
            if cut_type == "Circular Hole":
                hx = st.slider(
                    "X Position", 0.0, float(problem.width), float(problem.width / 2.0)
                )
                hy = st.slider(
                    "Y Position",
                    0.0,
                    float(problem.height),
                    float(problem.height / 2.0),
                )
                hr = st.slider(
                    "Radius", 1.0, float(min(problem.width, problem.height) / 3.0), 4.0
                )
                if st.button("➕ Cut Circle"):
                    human_model.cut_circle(hx, hy, hr)
                    st.rerun()
            else:
                rx = st.slider("X Position", 0.0, float(problem.width - 5.0), 10.0)
                ry = st.slider("Y Position", 0.0, float(problem.height - 5.0), 5.0)
                rw = st.slider("Width", 2.0, float(problem.width / 2.0), 10.0)
                rh = st.slider("Height", 2.0, float(problem.height / 2.0), 5.0)
                if st.button("➕ Cut Rectangle"):
                    human_model.cut_rect(rx, ry, rw, rh)
                    st.rerun()

            if st.button("↩️ Undo Last Cut"):
                human_model.undo()
                st.rerun()

        with col_view:
            fea_human = solver.solve(human_model.to_density_grid())
            fig = render_component_plot(
                human_model.to_density_grid(),
                f"Human CAD Component (Actions: {len(human_model.history)})",
                f"Compliance: {fea_human.compliance:.2f} | Volume: {human_model.get_volume_fraction():.2f}",
                stress=fea_human.von_mises_stress,
            )
            st.pyplot(fig)

    with tab_llm_log:
        st.subheader("🧠 LLM Designer Agent Reasoning Transcript")
        if "llm_res" in st.session_state:
            llm_res = st.session_state["llm_res"]
            for step_rec in llm_res.transcript:
                with st.expander(
                    f"Step {step_rec.step}: {step_rec.action}", expanded=True
                ):
                    st.markdown(f"**Thought**: {step_rec.thought}")
                    st.caption(
                        f"Compliance: {step_rec.compliance:.3f} | Volume Fraction: {step_rec.volume_fraction:.2f}"
                    )
        else:
            st.info(
                "Run the benchmark in the 4-Way Arena tab to view the LLM reasoning transcript."
            )

    with tab_export:
        st.subheader("💾 CAD File Export")
        st.write(
            "Export any component design to standard manufacturing formats (DXF / SVG)."
        )
        model_to_export = st.selectbox(
            "Select Design to Export",
            ["LLM CAD Model", "Dedicated Model", "Human CAD Model"],
        )
        target = None
        if model_to_export == "LLM CAD Model" and "llm_res" in st.session_state:
            target = st.session_state["llm_res"].cad_model
        elif model_to_export == "Dedicated Model" and "heur_model" in st.session_state:
            target = st.session_state["heur_model"]
        elif model_to_export == "Human CAD Model":
            target = st.session_state["human_model"]

        if target is not None:
            svg_content = export_cad_to_svg(target)
            dxf_content = export_cad_to_dxf(target)

            st.download_button(
                "⬇️ Download SVG Vector",
                svg_content,
                file_name=f"{problem.name.lower().replace(' ', '_')}.svg",
                mime="image/svg+xml",
            )
            st.download_button(
                "⬇️ Download DXF (AutoCAD R12)",
                dxf_content,
                file_name=f"{problem.name.lower().replace(' ', '_')}.dxf",
                mime="application/dxf",
            )
        else:
            st.warning("Run the benchmark first to enable model export.")


if __name__ == "__main__":
    main()
