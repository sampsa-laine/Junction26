import json
from pathlib import Path

import streamlit as st

from src.agents.csg import CADModel
from src.agents.export import export_cad_to_dxf, export_cad_to_svg
from src.agents.llm_designer import LLMStepRecord
from src.agents.problem import DesignProblem
from src.agents.solutions import (
    PROBLEM_PRESETS,
    InRepoSolverName,
    SolutionRecord,
    build_comparison_rows,
    build_human_solution,
    discover_harbor_runs,
    get_cached_reference_simp,
    load_design_payload_solution,
    load_design_solution,
    load_harbor_trial,
    replay_cutout_history,
    run_in_repo_solver,
)
from src.agents.visualizer import (
    render_replay_trajectory_figure,
    render_solution_views,
)

_SOLVER_LABELS: dict[InRepoSolverName, str] = {
    "simp": "SIMP (Continuous Optimum)",
    "oracle_solver": "Oracle Solver (SIMP-to-CAD)",
    "heuristic_agent": "Heuristic Agent (Low-Strain Cuts)",
    "llm_designer": "LLM Designer (ReAct Agent)",
}


def _problem_signature(problem: DesignProblem) -> str:
    return json.dumps(problem.to_dict(), sort_keys=True)


def _clamp_load_case_index(
    problem: DesignProblem, requested_index: int | None
) -> int | None:
    if requested_index is None:
        return None
    if 0 <= requested_index < len(problem.effective_load_cases):
        return requested_index
    return None


def _render_three_panel_views(
    solution_or_frame,
    problem: DesignProblem,
    load_case_index: int | None,
    log_strain: bool,
    highlight_cutout: bool = True,
) -> None:
    effective_case = _clamp_load_case_index(problem, load_case_index)
    views = render_solution_views(
        solution_or_frame,
        problem=problem,
        load_case_index=effective_case,
        log_strain=log_strain,
        highlight_cutout=highlight_cutout,
    )
    col_model, col_stress, col_strain = st.columns(3)
    with col_model:
        st.pyplot(views.model_figure)
    with col_stress:
        st.pyplot(views.stress_figure)
    with col_strain:
        st.pyplot(views.strain_figure)


def main() -> None:
    st.set_page_config(
        page_title="2D CAD TopOpt Inspector & Arena", page_icon="🏗️", layout="wide"
    )

    st.title("🏗️ 2D CAD Topology Optimization Evaluation & Replay Inspector")
    st.caption(
        "Compare Harbor benchmark runs and in-repo solvers with FEA stress/strain fields and step-by-step CSG cutout replay"
    )

    if "loaded_solutions" not in st.session_state:
        st.session_state["loaded_solutions"] = {}

    st.sidebar.header("1. Design Problem")
    preset_keys = list(PROBLEM_PRESETS.keys())
    selected_preset_key = st.sidebar.selectbox(
        "Benchmark Task / Preset",
        options=preset_keys,
        format_func=lambda key: PROBLEM_PRESETS[key].display_name,
        key="preset_key",
    )
    preset = PROBLEM_PRESETS[selected_preset_key]
    problem = preset.builder()

    override_vol = st.sidebar.checkbox(
        "Override Target Volume Fraction",
        value=False,
        key="override_target_volume",
    )
    if override_vol:
        problem.target_volume_fraction = float(
            st.sidebar.slider(
                "Custom Target Volume Fraction",
                min_value=0.20,
                max_value=0.70,
                value=float(problem.target_volume_fraction),
                step=0.01,
                key="custom_target_volume",
            )
        )

    num_cases = len(problem.effective_load_cases)
    stress_note = (
        f"<= {problem.max_stress_limit:.2f}"
        if problem.max_stress_limit is not None
        else "unconstrained"
    )
    st.sidebar.caption(
        f"Domain: {problem.width:.0f}x{problem.height:.0f} ({problem.nx}x{problem.ny}) | "
        f"Target Vol: {problem.target_volume_fraction:.2f} | "
        f"Stress: {stress_note} | Load Cases: {num_cases}"
    )

    prob_sig = _problem_signature(problem)
    if (
        "human_model" not in st.session_state
        or st.session_state.get("human_problem_sig") != prob_sig
    ):
        st.session_state["human_model"] = CADModel(problem)
        st.session_state["human_problem_sig"] = prob_sig

    st.sidebar.header("2. View & FEA Controls")
    max_cases = max(
        [
            num_cases,
            *(
                len(sol.problem.effective_load_cases)
                for sol in st.session_state["loaded_solutions"].values()
            ),
        ]
    )
    case_options: list[int | None] = [None, *range(max_cases)]
    if (
        "load_case_selector" in st.session_state
        and st.session_state["load_case_selector"] not in case_options
    ):
        st.session_state["load_case_selector"] = None
    selected_load_case = st.sidebar.selectbox(
        "Load Case View",
        options=case_options,
        format_func=lambda idx: (
            "All Cases (Envelope / Sum)" if idx is None else f"Load Case {idx + 1}"
        ),
        key="load_case_selector",
    )
    log_strain = st.sidebar.checkbox(
        "Log-Scale Strain Energy", value=False, key="log_strain"
    )

    st.sidebar.header("3. In-Repo Solvers")
    solver_keys: list[InRepoSolverName] = [
        "simp",
        "oracle_solver",
        "heuristic_agent",
        "llm_designer",
    ]
    selected_solvers = st.sidebar.multiselect(
        "Solvers to Run",
        options=solver_keys,
        default=solver_keys,
        format_func=lambda key: _SOLVER_LABELS[key],
        key="selected_solvers",
    )
    if st.sidebar.button(
        "🚀 Run In-Repo Solvers", type="primary", key="run_solvers_btn"
    ):
        ref_simp = None
        if "simp" in selected_solvers:
            simp_record = run_in_repo_solver(problem, "simp")
            st.session_state["loaded_solutions"][simp_record.solution_id] = simp_record
            ref_simp = get_cached_reference_simp(problem)
        else:
            ref_simp = get_cached_reference_simp(problem)

        for s_key in selected_solvers:
            if s_key == "simp":
                continue
            sol_record = run_in_repo_solver(problem, s_key, reference_simp=ref_simp)
            st.session_state["loaded_solutions"][sol_record.solution_id] = sol_record

    st.sidebar.header("4. Harbor Runs & Design Files")
    discovered_paths = discover_harbor_runs("jobs")
    filter_to_task = st.sidebar.checkbox(
        "Filter Harbor runs to current task",
        value=False,
        key="filter_harbor_to_task",
    )
    if filter_to_task and preset.task_id is not None:
        filtered_paths = [
            p
            for p in discovered_paths
            if p.name.startswith(preset.task_id) or preset.task_id in str(p)
        ]
    else:
        filtered_paths = discovered_paths

    trial_option_map = {str(p): p for p in filtered_paths}
    if "selected_harbor_trials" in st.session_state:
        st.session_state["selected_harbor_trials"] = [
            s
            for s in st.session_state["selected_harbor_trials"]
            if s in trial_option_map
        ]
    selected_trial_strs = st.sidebar.multiselect(
        "Discovered Harbor Trials (jobs/)",
        options=list(trial_option_map.keys()),
        format_func=lambda s: f"{Path(s).parent.name}/{Path(s).name}",
        key="selected_harbor_trials",
    )
    if st.sidebar.button("📥 Load Selected Harbor Trials", key="load_harbor_btn"):
        for t_str in selected_trial_strs:
            try:
                rec = load_harbor_trial(trial_option_map[t_str])
                st.session_state["loaded_solutions"][rec.solution_id] = rec
            except Exception as exc:
                st.sidebar.error(f"Failed to load {t_str}: {exc}")

    custom_path_str = st.sidebar.text_input(
        "Custom Trial Dir or design.json Path",
        value="",
        key="custom_solution_path",
    )
    if st.sidebar.button("📂 Load Custom Path", key="load_custom_path_btn"):
        cleaned_path = custom_path_str.strip()
        if cleaned_path:
            c_path = Path(cleaned_path)
            try:
                if c_path.is_dir():
                    trials = discover_harbor_runs(c_path)
                    targets = trials if trials else [c_path]
                    for target_dir in targets:
                        rec = load_harbor_trial(target_dir)
                        st.session_state["loaded_solutions"][rec.solution_id] = rec
                else:
                    try:
                        rec = load_design_solution(c_path)
                    except ValueError:
                        rec = load_design_solution(c_path, problem=problem)
                    st.session_state["loaded_solutions"][rec.solution_id] = rec
            except Exception as exc:
                st.sidebar.error(f"Could not load {cleaned_path}: {exc}")

    uploaded_file = st.sidebar.file_uploader(
        "Upload design.json", type=["json"], key="upload_design_json"
    )
    if uploaded_file is not None:
        upload_key = f"uploaded:{uploaded_file.name}:{uploaded_file.size}"
        if upload_key not in st.session_state["loaded_solutions"]:
            try:
                payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
                try:
                    rec = load_design_payload_solution(
                        payload, label=f"Uploaded ({uploaded_file.name})"
                    )
                except ValueError:
                    rec = load_design_payload_solution(
                        payload,
                        problem=problem,
                        label=f"Uploaded ({uploaded_file.name})",
                    )
                rec.solution_id = upload_key
                st.session_state["loaded_solutions"][upload_key] = rec
            except Exception as exc:
                st.sidebar.error(f"Invalid design.json upload: {exc}")

    if st.sidebar.button("🗑️ Clear Loaded Solutions", key="clear_solutions_btn"):
        st.session_state["loaded_solutions"] = {}
        st.session_state.pop("active_replay_model", None)
        st.session_state.pop("active_replay_label", None)

    updated_max_cases = max(
        [
            num_cases,
            *(
                len(sol.problem.effective_load_cases)
                for sol in st.session_state["loaded_solutions"].values()
            ),
        ]
    )
    if updated_max_cases != max_cases:
        st.rerun()

    loaded_solutions: dict[str, SolutionRecord] = st.session_state["loaded_solutions"]
    human_model: CADModel = st.session_state["human_model"]

    tab_compare, tab_replay, tab_human, tab_export = st.tabs(
        [
            "🔍 Solution Comparison & Inspection",
            "⏪ Cutout History Replay",
            "✏️ Human Canvas & LLM Log",
            "💾 CAD Export",
        ]
    )

    with tab_compare:
        only_current = st.checkbox(
            "Filter displayed solutions to selected problem",
            value=False,
            key="filter_compare_current_problem",
        )
        include_human = st.checkbox(
            "Include Human Designer Canvas",
            value=len(human_model.history) > 0,
            key="include_human_compare",
        )

        displayed_solutions: list[SolutionRecord] = [
            sol
            for sol in loaded_solutions.values()
            if not only_current or sol.problem.name == problem.name
        ]
        if include_human:
            displayed_solutions.append(build_human_solution(problem, human_model))

        if displayed_solutions:
            st.subheader("📊 Unified Comparison Scoreboard")
            rows = build_comparison_rows(displayed_solutions)
            st.table(rows)

            st.subheader("🔬 Resulting Model, Stress & Strain Inspection")
            for sol in displayed_solutions:
                with st.container():
                    st.markdown(f"#### {sol.label} (`{sol.problem.name}`)")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(
                        "Harbor Reward",
                        f"{sol.harbor_reward:.4f}"
                        if sol.harbor_reward is not None
                        else "N/A",
                    )
                    m2.metric("Arena Score", f"{sol.scoreboard.overall_score:.1f}/100")
                    m3.metric("Compliance", f"{sol.fea_result.compliance:.2f}")
                    m4.metric(
                        "Volume / Target",
                        f"{sol.scoreboard.volume_fraction:.3f} / {sol.problem.target_volume_fraction:.2f}",
                    )
                    max_vm = float(sol.fea_result.von_mises_stress.max())
                    stress_limit = sol.problem.max_stress_limit
                    m5.metric(
                        "Max Stress",
                        f"{max_vm:.3f}"
                        + (
                            f" / {stress_limit:.2f}" if stress_limit is not None else ""
                        ),
                    )

                    _render_three_panel_views(
                        sol,
                        problem=sol.problem,
                        load_case_index=selected_load_case,
                        log_strain=log_strain,
                        highlight_cutout=False,
                    )
                    st.divider()
        else:
            st.info(
                "Use the sidebar to run in-repo solvers (`SIMP`, `Oracle`, `Heuristic`, `LLM`) "
                "or load Harbor benchmark trials (`jobs/`) and `design.json` artifacts."
            )

    with tab_replay:
        st.subheader("⏪ Step-by-Step Saved Cutout History Replay (0..N)")
        replay_candidates: dict[str, SolutionRecord] = dict(loaded_solutions)
        if not replay_candidates or len(human_model.history) > 0:
            human_sol = build_human_solution(problem, human_model)
            replay_candidates[human_sol.solution_id] = human_sol

        candidate_ids = list(replay_candidates.keys())
        default_idx = 0
        for idx, cid in enumerate(candidate_ids):
            cand = replay_candidates[cid]
            if cand.cad_model is not None and len(cand.cad_model.history) > 0:
                default_idx = idx
                break

        replay_option_labels = {
            cid: (
                f"{rec.label} ({len(rec.cad_model.history)} cutouts)"
                if rec.cad_model is not None
                else f"{rec.label} (Continuous Density)"
            )
            for cid, rec in replay_candidates.items()
        }

        if (
            "replay_solution_selector" in st.session_state
            and st.session_state["replay_solution_selector"] not in replay_candidates
        ):
            st.session_state["replay_solution_selector"] = candidate_ids[default_idx]

        selected_replay_id = st.selectbox(
            "Select Solution to Replay",
            options=candidate_ids,
            index=default_idx,
            format_func=lambda cid: replay_option_labels.get(str(cid), str(cid)),
            key="replay_solution_selector",
        )
        replay_sol = replay_candidates[selected_replay_id]

        if replay_sol.cad_model is None:
            st.session_state.pop("active_replay_model", None)
            st.session_state.pop("active_replay_label", None)
            st.session_state.pop("replay_step_slider", None)
            st.session_state.pop("_replay_slider_sol_id", None)
            st.session_state.pop("_replay_slider_max", None)
            st.info(
                "Continuous SIMP density solutions do not contain parametric CSG `CutoutAction` history. "
                "Select a parametric `CADModel` solution (Harbor trial, Oracle Solver, Heuristic Agent, "
                "LLM Designer, or `design.json`) to step through cutouts `0..N`."
            )
        else:
            cad_target = replay_sol.cad_model
            total_steps = len(cad_target.history)
            effective_replay_case = _clamp_load_case_index(
                replay_sol.problem, selected_load_case
            )

            if total_steps == 0:
                active_step = 0
                st.session_state.pop("replay_step_slider", None)
                st.session_state.pop("_replay_slider_sol_id", None)
                st.session_state.pop("_replay_slider_max", None)
                st.info(
                    "Step 0 / 0: This `CADModel` has 0 cutouts (uncut initial stock domain with keepouts applied)."
                )
            else:
                prev_sol_id = st.session_state.get("_replay_slider_sol_id")
                prev_max = st.session_state.get("_replay_slider_max")
                if (
                    prev_sol_id != selected_replay_id
                    or "replay_step_slider" not in st.session_state
                ):
                    st.session_state["replay_step_slider"] = total_steps
                elif (
                    isinstance(prev_max, int)
                    and total_steps > prev_max
                    and st.session_state["replay_step_slider"] == prev_max
                ):
                    st.session_state["replay_step_slider"] = total_steps
                else:
                    clamped_step = max(
                        0, min(int(st.session_state["replay_step_slider"]), total_steps)
                    )
                    if st.session_state["replay_step_slider"] != clamped_step:
                        st.session_state["replay_step_slider"] = clamped_step
                st.session_state["_replay_slider_sol_id"] = selected_replay_id
                st.session_state["_replay_slider_max"] = total_steps
                active_step = int(
                    st.slider(
                        "Replay Step (0 = Uncut Stock Domain, N = Final Design)",
                        min_value=0,
                        max_value=total_steps,
                        step=1,
                        key="replay_step_slider",
                    )
                )

            frames = replay_cutout_history(
                cad_target, load_case_index=effective_replay_case
            )
            current_frame = frames[active_step]
            st.session_state["active_replay_model"] = current_frame.cad_model
            st.session_state["active_replay_label"] = (
                f"{replay_sol.label} (Step {active_step}/{total_steps})"
            )

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Step", f"{active_step} / {total_steps}")
            r2.metric(
                "Volume Fraction",
                f"{current_frame.volume_fraction:.4f} (target <= {replay_sol.problem.target_volume_fraction:.2f})",
            )
            r3.metric("Compliance", f"{current_frame.compliance:.2f}")
            r4.metric(
                "Max Von Mises / Connected",
                f"{current_frame.max_von_mises_stress:.3f} | {'Yes' if current_frame.structurally_connected else 'No'}",
            )

            if current_frame.action is None:
                st.caption(
                    "**Step 0**: Initial stock domain prior to any cutouts (`VOID` keepouts applied)."
                )
            else:
                act = current_frame.action
                mirror_tags: list[str] = []
                if act.mirror_x:
                    mirror_tags.append("mirror_x")
                if act.mirror_y:
                    mirror_tags.append("mirror_y")
                mirror_str = (
                    f" | Mirrors: {', '.join(mirror_tags)}" if mirror_tags else ""
                )
                st.caption(
                    f"**Step {active_step} Cutout (`{act.action_type}`)**: "
                    f"`params={tuple(round(v, 3) for v in act.params)}`{mirror_str} — {act.description}"
                )

            _render_three_panel_views(
                current_frame,
                problem=replay_sol.problem,
                load_case_index=effective_replay_case,
                log_strain=log_strain,
                highlight_cutout=True,
            )

            if len(frames) > 1:
                st.markdown("##### Cutout History Trajectory (0..N)")
                traj_fig = render_replay_trajectory_figure(
                    frames, replay_sol.problem, active_step=active_step
                )
                st.pyplot(traj_fig)

            step_rows = [
                {
                    "Step": f.step,
                    "Action": f.action.action_type if f.action else "stock",
                    "Description": f.action.description
                    if f.action
                    else "Initial stock domain",
                    "Volume": f"{f.volume_fraction:.4f}",
                    "Compliance": f"{f.compliance:.2f}",
                    "Max Stress": f"{f.max_von_mises_stress:.3f}",
                    "Connected": "Yes" if f.structurally_connected else "No",
                }
                for f in frames
            ]
            st.table(step_rows)

    with tab_human:
        st.subheader("✏️ Interactive Human Designer Canvas")
        col_tools, col_view = st.columns([1, 3])
        with col_tools:
            cut_type = st.radio(
                "Cutout Primitive",
                ["Circular Hole", "Rectangular Cutout", "Slot Cutout"],
                key="human_cut_type",
            )
            if cut_type == "Circular Hole":
                hx = st.slider(
                    "Center X",
                    0.0,
                    float(problem.width),
                    float(problem.width / 2.0),
                    key="human_cx",
                )
                hy = st.slider(
                    "Center Y",
                    0.0,
                    float(problem.height),
                    float(problem.height / 2.0),
                    key="human_cy",
                )
                hr = st.slider(
                    "Radius",
                    float(problem.min_feature_radius),
                    float(min(problem.width, problem.height) / 3.0),
                    4.0,
                    key="human_cr",
                )
                if st.button("➕ Cut Circle", key="human_cut_circle_btn"):
                    if human_model.cut_circle(hx, hy, hr):
                        st.rerun()
                    else:
                        st.warning(
                            "Cutout rejected (outside material or inside solid keepout)."
                        )
            elif cut_type == "Rectangular Cutout":
                rx = st.slider(
                    "Corner X", 0.0, float(problem.width - 4.0), 10.0, key="human_rx"
                )
                ry = st.slider(
                    "Corner Y", 0.0, float(problem.height - 4.0), 5.0, key="human_ry"
                )
                rw = st.slider(
                    "Width", 3.0, float(problem.width / 2.0), 10.0, key="human_rw"
                )
                rh = st.slider(
                    "Height", 3.0, float(problem.height / 2.0), 5.0, key="human_rh"
                )
                rfillet = st.slider("Corner Fillet", 0.0, 4.0, 1.0, key="human_rf")
                if st.button("➕ Cut Rectangle", key="human_cut_rect_btn"):
                    if human_model.cut_rect(rx, ry, rw, rh, fillet_radius=rfillet):
                        st.rerun()
                    else:
                        st.warning(
                            "Cutout rejected (outside material or inside solid keepout)."
                        )
            else:
                sx1 = st.slider(
                    "Start X", 0.0, float(problem.width), 15.0, key="human_sx1"
                )
                sy1 = st.slider(
                    "Start Y", 0.0, float(problem.height), 15.0, key="human_sy1"
                )
                sx2 = st.slider(
                    "End X", 0.0, float(problem.width), 35.0, key="human_sx2"
                )
                sy2 = st.slider(
                    "End Y", 0.0, float(problem.height), 15.0, key="human_sy2"
                )
                sr = st.slider(
                    "Slot Radius",
                    float(problem.min_feature_radius),
                    6.0,
                    2.5,
                    key="human_sr",
                )
                if st.button("➕ Cut Slot", key="human_cut_slot_btn"):
                    if human_model.cut_slot(sx1, sy1, sx2, sy2, sr):
                        st.rerun()
                    else:
                        st.warning(
                            "Cutout rejected (degenerate slot, outside material, or inside solid keepout)."
                        )

            c_undo, c_reset = st.columns(2)
            if c_undo.button("↩️ Undo", key="human_undo_btn"):
                if human_model.undo():
                    st.rerun()
            if c_reset.button("🔄 Reset", key="human_reset_btn"):
                if len(human_model.history) > 0:
                    human_model.reset()
                    st.rerun()

        with col_view:
            human_sol_view = build_human_solution(problem, human_model)
            st.caption(
                f"Cutouts: {len(human_model.history)} | "
                f"Volume: {human_model.get_volume_fraction():.3f} | "
                f"Compliance: {human_sol_view.fea_result.compliance:.2f} | "
                f"Harbor Reward: {human_sol_view.harbor_reward:.4f}"
            )
            _render_three_panel_views(
                human_sol_view,
                problem=problem,
                load_case_index=selected_load_case,
                log_strain=log_strain,
                highlight_cutout=False,
            )

        st.divider()
        st.subheader("🧠 LLM Designer Reasoning Transcript")
        llm_records = [
            sol
            for sol in loaded_solutions.values()
            if sol.solver_name == "llm_designer" and sol.metadata.get("transcript")
        ]
        if llm_records:
            latest_llm = llm_records[-1]
            transcript: list[LLMStepRecord] = latest_llm.metadata["transcript"]
            for step_rec in transcript:
                with st.expander(
                    f"Step {step_rec.step}: {step_rec.action}", expanded=True
                ):
                    st.markdown(f"**Thought**: {step_rec.thought}")
                    st.caption(
                        f"Compliance: {step_rec.compliance:.3f} | Volume Fraction: {step_rec.volume_fraction:.3f}"
                    )
        else:
            st.info(
                "Run `LLM Designer` from the sidebar to inspect its step-by-step ReAct reasoning transcript."
            )

    with tab_export:
        st.subheader("💾 CAD Vector & Artifact Export")
        exportable_models: dict[str, CADModel] = {}
        cad_solutions = [
            sol for sol in loaded_solutions.values() if sol.cad_model is not None
        ]
        label_counts: dict[str, int] = {}
        distinct_problems: set[str] = set()
        for sol in cad_solutions:
            label_counts[sol.label] = label_counts.get(sol.label, 0) + 1
            distinct_problems.add(sol.problem.name)
        for sol in cad_solutions:
            assert sol.cad_model is not None
            base_export_key = (
                f"{sol.label} ({sol.problem.name})"
                if label_counts[sol.label] > 1 or len(distinct_problems) > 1
                else sol.label
            )
            export_key = base_export_key
            suffix = 2
            while export_key in exportable_models:
                export_key = f"{base_export_key} [{suffix}]"
                suffix += 1
            exportable_models[export_key] = sol.cad_model
        if "active_replay_model" in st.session_state:
            replay_lbl = st.session_state.get(
                "active_replay_label", "Active Replayed Step"
            )
            exportable_models[f"Replay: {replay_lbl}"] = st.session_state[
                "active_replay_model"
            ]
        exportable_models["Human Designer Canvas"] = human_model
        export_keys = list(exportable_models.keys())
        if (
            "export_model_selector" in st.session_state
            and st.session_state["export_model_selector"] not in exportable_models
        ):
            st.session_state["export_model_selector"] = export_keys[0]

        chosen_export_label = st.selectbox(
            "Select Parametric CADModel to Export",
            options=export_keys,
            key="export_model_selector",
        )
        export_target = exportable_models[chosen_export_label]
        slug = export_target.problem.name.lower().replace(" ", "_")

        svg_str = export_cad_to_svg(export_target)
        dxf_str = export_cad_to_dxf(export_target)
        json_str = json.dumps(export_target.to_dict(), indent=2)

        e1, e2, e3 = st.columns(3)
        e1.download_button(
            "⬇️ Download SVG Vector",
            svg_str,
            file_name=f"{slug}.svg",
            mime="image/svg+xml",
        )
        e2.download_button(
            "⬇️ Download DXF (AutoCAD R12)",
            dxf_str,
            file_name=f"{slug}.dxf",
            mime="application/dxf",
        )
        e3.download_button(
            "⬇️ Download design.json",
            json_str,
            file_name="design.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
