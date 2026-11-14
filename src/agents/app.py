from collections.abc import Sequence
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import streamlit as st

from src.agents.csg import CADModel
from src.agents.export import export_cad_to_dxf, export_cad_to_svg
from src.agents.llm_designer import LLMStepRecord
from src.agents.problem import DesignProblem
from src.agents.solutions import (
    PROBLEM_PRESETS,
    InRepoSolverName,
    ReplayFrame,
    SolutionRecord,
    _problem_short_id,
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
    compute_load_case_fea,
    render_fea_field_figure,
    render_model_geometry_figure,
    render_replay_trajectory_figure,
    render_solution_views,
)

_SOLVER_LABELS: dict[InRepoSolverName, str] = {
    "simp": "SIMP (Continuous Optimum)",
    "oracle_solver": "Oracle Solver (SIMP-to-CAD)",
    "heuristic_agent": "Heuristic Agent (Low-Strain Cuts)",
    "llm_designer": "LLM Designer (ReAct Agent)",
}

_HUMAN_SLIDER_KEYS = (
    "human_cx",
    "human_cy",
    "human_cr",
    "human_rx",
    "human_ry",
    "human_rw",
    "human_rh",
    "human_rf",
    "human_sx1",
    "human_sy1",
    "human_sx2",
    "human_sy2",
    "human_sr",
)


def _problem_signature(problem: DesignProblem) -> str:
    return json.dumps(problem.to_dict(), sort_keys=True)


def _history_signature(model: CADModel | None) -> str:
    if model is None:
        return "continuous"
    return json.dumps([a.to_dict() for a in model.history], sort_keys=True)


def _figure_to_png_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def _get_cached_human_solution(
    problem: DesignProblem, human_model: CADModel
) -> SolutionRecord:
    cache: dict[tuple[str, str], SolutionRecord] = st.session_state.setdefault(
        "_human_solution_cache", {}
    )
    key = (_problem_signature(problem), _history_signature(human_model))
    cached = cache.get(key)
    if cached is not None:
        return cached
    sol = build_human_solution(problem, human_model)
    cache[key] = sol
    return sol


def _get_cached_replay_frames(
    solution_id: str, cad_model: CADModel, load_case_index: int | None
) -> list[ReplayFrame]:
    cache: dict[tuple[str, str, str, int | None], list[ReplayFrame]] = (
        st.session_state.setdefault("_replay_frames_cache", {})
    )
    key = (
        solution_id,
        _problem_signature(cad_model.problem),
        _history_signature(cad_model),
        load_case_index,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    frames = replay_cutout_history(cad_model, load_case_index=load_case_index)
    cache[key] = frames
    return frames


def _clamp_load_case_index(
    problem: DesignProblem, requested_index: int | None
) -> int | None:
    if requested_index is None:
        return None
    if 0 <= requested_index < len(problem.effective_load_cases):
        return requested_index
    return None


def _preferred_replay_solution_id(
    solutions: Sequence[SolutionRecord], fallback_id: str | None = None
) -> str | None:
    for sol in solutions:
        if sol.cad_model is not None and len(sol.cad_model.history) > 0:
            return sol.solution_id
    for sol in solutions:
        if sol.cad_model is not None:
            return sol.solution_id
    if solutions:
        return solutions[0].solution_id
    return fallback_id


def _set_active_replay_selection(solution_id: str | None) -> None:
    if solution_id is None:
        return
    if st.session_state.get("replay_solution_selector") != solution_id:
        st.session_state["replay_solution_selector"] = solution_id
        st.session_state.pop("active_replay_model", None)
        st.session_state.pop("active_replay_label", None)
        st.session_state.pop("replay_step_slider", None)
        st.session_state.pop("_replay_slider_sol_id", None)
        st.session_state.pop("_replay_slider_max", None)


def _render_stock_problem_preview(
    problem: DesignProblem,
    load_case_index: int | None,
    log_strain: bool,
) -> None:
    effective_case = _clamp_load_case_index(problem, load_case_index)
    png_cache: dict[tuple[Any, ...], tuple[bytes, bytes, bytes]] = (
        st.session_state.setdefault("_plot_png_cache", {})
    )
    cache_key = ("stock", _problem_signature(problem), effective_case, log_strain)
    cached_pngs = png_cache.get(cache_key)
    if cached_pngs is None:
        stock_model = CADModel(problem)
        stock_density = stock_model.to_density_grid()
        stock_fea = compute_load_case_fea(
            problem, stock_density, load_case_index=effective_case, penalty=1.0
        )
        case_tag = (
            f" [Case {effective_case + 1}]"
            if effective_case is not None
            else (" [All Cases]" if len(problem.effective_load_cases) > 1 else "")
        )
        cached_pngs = (
            _figure_to_png_bytes(
                render_model_geometry_figure(
                    problem=problem,
                    cad_model=stock_model,
                    density=stock_density,
                    load_case_index=effective_case,
                    title=f"{problem.name} — Stock Geometry{case_tag}",
                )
            ),
            _figure_to_png_bytes(
                render_fea_field_figure(
                    problem=problem,
                    density=stock_density,
                    fea_result=stock_fea,
                    field="stress",
                    cad_model=stock_model,
                    log_scale=False,
                    title=f"Stock Von Mises Stress{case_tag} (Max={float(stock_fea.von_mises_stress.max()):.3f})",
                )
            ),
            _figure_to_png_bytes(
                render_fea_field_figure(
                    problem=problem,
                    density=stock_density,
                    fea_result=stock_fea,
                    field="strain_energy",
                    cad_model=stock_model,
                    log_scale=log_strain,
                    title=f"Stock Strain Energy{case_tag} (C={stock_fea.compliance:.2f})",
                )
            ),
        )
        png_cache[cache_key] = cached_pngs

    col_model, col_stress, col_strain = st.columns(3)
    with col_model:
        st.image(cached_pngs[0], width="stretch")
    with col_stress:
        st.image(cached_pngs[1], width="stretch")
    with col_strain:
        st.image(cached_pngs[2], width="stretch")


def _render_three_panel_views(
    solution_or_frame,
    problem: DesignProblem,
    load_case_index: int | None,
    log_strain: bool,
    highlight_cutout: bool = True,
    cache_key: tuple[Any, ...] | None = None,
) -> None:
    effective_case = _clamp_load_case_index(problem, load_case_index)
    png_cache: dict[tuple[Any, ...], tuple[bytes, bytes, bytes]] = (
        st.session_state.setdefault("_plot_png_cache", {})
    )
    cached_pngs = png_cache.get(cache_key) if cache_key is not None else None
    if cached_pngs is None:
        views = render_solution_views(
            solution_or_frame,
            problem=problem,
            load_case_index=effective_case,
            log_strain=log_strain,
            highlight_cutout=highlight_cutout,
        )
        cached_pngs = (
            _figure_to_png_bytes(views.model_figure),
            _figure_to_png_bytes(views.stress_figure),
            _figure_to_png_bytes(views.strain_figure),
        )
        if cache_key is not None:
            png_cache[cache_key] = cached_pngs

    col_model, col_stress, col_strain = st.columns(3)
    with col_model:
        st.image(cached_pngs[0], width="stretch")
    with col_stress:
        st.image(cached_pngs[1], width="stretch")
    with col_strain:
        st.image(cached_pngs[2], width="stretch")


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
    if "_cleared_upload_keys" not in st.session_state:
        st.session_state["_cleared_upload_keys"] = set()

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

    prev_preset_key = st.session_state.get("_prev_preset_key")
    if prev_preset_key is not None and prev_preset_key != selected_preset_key:
        st.session_state.pop("custom_target_volume", None)
    st.session_state["_prev_preset_key"] = selected_preset_key

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
    prev_prob_sig = st.session_state.get("human_problem_sig")
    problem_changed = prev_prob_sig is not None and prev_prob_sig != prob_sig
    if "human_model" not in st.session_state or problem_changed:
        st.session_state["human_model"] = CADModel(problem)
        st.session_state["human_problem_sig"] = prob_sig
        for slider_key in _HUMAN_SLIDER_KEYS:
            st.session_state.pop(slider_key, None)
        if problem_changed:
            st.session_state["filter_compare_current_problem"] = True
            st.session_state["include_human_compare"] = False
            matching_loaded = [
                sol
                for sol in st.session_state["loaded_solutions"].values()
                if _problem_signature(sol.problem) == prob_sig
            ]
            _set_active_replay_selection(
                _preferred_replay_solution_id(
                    matching_loaded,
                    fallback_id=f"human:{_problem_short_id(problem)}",
                )
            )
            st.session_state.pop("export_model_selector", None)

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
        stale_solver_ids = [
            sid
            for sid, sol in st.session_state["loaded_solutions"].items()
            if sol.source_kind == "in_repo_solver"
            and _problem_signature(sol.problem) == prob_sig
        ]
        for sid in stale_solver_ids:
            st.session_state["loaded_solutions"].pop(sid, None)

        newly_run_records: list[SolutionRecord] = []
        if "simp" in selected_solvers:
            simp_record = run_in_repo_solver(problem, "simp")
            st.session_state["loaded_solutions"][simp_record.solution_id] = simp_record
            newly_run_records.append(simp_record)
            ref_simp = get_cached_reference_simp(problem)
        else:
            ref_simp = get_cached_reference_simp(problem)

        for s_key in selected_solvers:
            if s_key == "simp":
                continue
            sol_record = run_in_repo_solver(problem, s_key, reference_simp=ref_simp)
            st.session_state["loaded_solutions"][sol_record.solution_id] = sol_record
            newly_run_records.append(sol_record)

        if newly_run_records:
            _set_active_replay_selection(
                _preferred_replay_solution_id(newly_run_records)
            )
            st.session_state.pop("export_model_selector", None)

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
    newly_loaded_external: list[SolutionRecord] = []
    if st.sidebar.button("📥 Load Selected Harbor Trials", key="load_harbor_btn"):
        for t_str in selected_trial_strs:
            try:
                rec = load_harbor_trial(trial_option_map[t_str])
                st.session_state["loaded_solutions"].pop(rec.solution_id, None)
                st.session_state["loaded_solutions"][rec.solution_id] = rec
                newly_loaded_external.append(rec)
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
                        st.session_state["loaded_solutions"].pop(rec.solution_id, None)
                        st.session_state["loaded_solutions"][rec.solution_id] = rec
                        newly_loaded_external.append(rec)
                else:
                    try:
                        rec = load_design_solution(c_path)
                    except ValueError:
                        rec = load_design_solution(c_path, problem=problem)
                    st.session_state["loaded_solutions"].pop(rec.solution_id, None)
                    st.session_state["loaded_solutions"][rec.solution_id] = rec
                    newly_loaded_external.append(rec)
            except Exception as exc:
                st.sidebar.error(f"Could not load {cleaned_path}: {exc}")

    uploaded_file = st.sidebar.file_uploader(
        "Upload design.json", type=["json"], key="upload_design_json"
    )
    active_upload_key: str | None = None
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        content_digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
        active_upload_key = f"uploaded:{uploaded_file.name}:{content_digest}"
        if (
            active_upload_key not in st.session_state["loaded_solutions"]
            and active_upload_key not in st.session_state["_cleared_upload_keys"]
        ):
            try:
                payload = json.loads(raw_bytes.decode("utf-8"))
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
                rec.solution_id = active_upload_key
                st.session_state["loaded_solutions"][active_upload_key] = rec
                newly_loaded_external.append(rec)
            except Exception as exc:
                st.sidebar.error(f"Invalid design.json upload: {exc}")

    if newly_loaded_external:
        if any(
            _problem_signature(rec.problem) != prob_sig for rec in newly_loaded_external
        ):
            st.session_state["filter_compare_current_problem"] = False
        _set_active_replay_selection(
            _preferred_replay_solution_id(newly_loaded_external)
        )
        st.session_state.pop("export_model_selector", None)

    if st.sidebar.button("🗑️ Clear Loaded Solutions", key="clear_solutions_btn"):
        if active_upload_key is not None:
            st.session_state["_cleared_upload_keys"].add(active_upload_key)
        st.session_state["loaded_solutions"] = {}
        st.session_state.pop("_replay_frames_cache", None)
        st.session_state.pop("_plot_png_cache", None)
        st.session_state.pop("active_replay_model", None)
        st.session_state.pop("active_replay_label", None)
        st.session_state.pop("replay_solution_selector", None)
        st.session_state.pop("export_model_selector", None)

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
        if st.session_state.pop("_enable_human_compare_next_run", False):
            st.session_state["include_human_compare"] = True
        only_current = st.checkbox(
            "Filter displayed solutions to selected problem",
            value=True,
            key="filter_compare_current_problem",
        )
        include_human = st.checkbox(
            "Include Human Designer Canvas",
            value=len(human_model.history) > 0,
            key="include_human_compare",
        )

        if only_current:
            displayed_solutions: list[SolutionRecord] = [
                sol
                for sol in loaded_solutions.values()
                if _problem_signature(sol.problem) == prob_sig
            ]
        else:
            current_problem_solutions = [
                sol
                for sol in loaded_solutions.values()
                if _problem_signature(sol.problem) == prob_sig
            ]
            other_problem_solutions = [
                sol
                for sol in loaded_solutions.values()
                if _problem_signature(sol.problem) != prob_sig
            ]
            displayed_solutions = [
                *current_problem_solutions,
                *other_problem_solutions,
            ]
        if include_human:
            displayed_solutions.append(_get_cached_human_solution(problem, human_model))

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
                        cache_key=(
                            "compare",
                            sol.solution_id,
                            _problem_signature(sol.problem),
                            _history_signature(sol.cad_model),
                            _clamp_load_case_index(sol.problem, selected_load_case),
                            log_strain,
                            False,
                        ),
                    )
                    st.divider()
        else:
            st.subheader(f"📐 Selected Problem Geometry (`{problem.name}`)")
            _render_stock_problem_preview(
                problem=problem,
                load_case_index=selected_load_case,
                log_strain=log_strain,
            )
            st.info(
                "Use the sidebar to run in-repo solvers (`SIMP`, `Oracle`, `Heuristic`, `LLM`) "
                "or load Harbor benchmark trials (`jobs/`) and `design.json` artifacts."
            )

    with tab_replay:
        st.subheader("⏪ Step-by-Step Saved Cutout History Replay (0..N)")
        replay_candidates: dict[str, SolutionRecord] = dict(loaded_solutions)
        has_current_problem_solution = any(
            _problem_signature(sol.problem) == prob_sig
            for sol in loaded_solutions.values()
        )
        if not has_current_problem_solution or len(human_model.history) > 0:
            human_sol = _get_cached_human_solution(problem, human_model)
            replay_candidates[human_sol.solution_id] = human_sol

        candidate_ids = list(replay_candidates.keys())
        matching_candidates = [
            replay_candidates[cid]
            for cid in candidate_ids
            if _problem_signature(replay_candidates[cid].problem) == prob_sig
        ]
        preferred_id = _preferred_replay_solution_id(
            matching_candidates
        ) or _preferred_replay_solution_id(list(replay_candidates.values()))
        default_idx = (
            candidate_ids.index(preferred_id) if preferred_id in candidate_ids else 0
        )

        distinct_replay_problems = {
            rec.problem.name for rec in replay_candidates.values()
        }
        replay_label_counts: dict[str, int] = {}
        for rec in replay_candidates.values():
            replay_label_counts[rec.label] = replay_label_counts.get(rec.label, 0) + 1

        replay_option_labels = {
            cid: (
                f"{rec.label}"
                f"{f' [{rec.problem.name}]' if len(distinct_replay_problems) > 1 or replay_label_counts[rec.label] > 1 else ''}"
                f" ({f'{len(rec.cad_model.history)} cutouts' if rec.cad_model is not None else 'Continuous Density'})"
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

            frames = _get_cached_replay_frames(
                selected_replay_id,
                cad_target,
                load_case_index=effective_replay_case,
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

            replay_prob_sig = _problem_signature(replay_sol.problem)
            replay_hist_sig = _history_signature(cad_target)
            _render_three_panel_views(
                current_frame,
                problem=replay_sol.problem,
                load_case_index=effective_replay_case,
                log_strain=log_strain,
                highlight_cutout=True,
                cache_key=(
                    "replay_frame",
                    selected_replay_id,
                    replay_prob_sig,
                    replay_hist_sig,
                    active_step,
                    effective_replay_case,
                    log_strain,
                    True,
                ),
            )

            if len(frames) > 1:
                st.markdown("##### Cutout History Trajectory (0..N)")
                png_cache: dict[tuple[Any, ...], Any] = st.session_state.setdefault(
                    "_plot_png_cache", {}
                )
                traj_key = (
                    "replay_traj",
                    selected_replay_id,
                    replay_prob_sig,
                    replay_hist_sig,
                    effective_replay_case,
                    active_step,
                )
                traj_png = png_cache.get(traj_key)
                if traj_png is None:
                    traj_png = _figure_to_png_bytes(
                        render_replay_trajectory_figure(
                            frames, replay_sol.problem, active_step=active_step
                        )
                    )
                    png_cache[traj_key] = traj_png
                st.image(traj_png, width="stretch")

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
                        st.session_state["_enable_human_compare_next_run"] = True
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
                        st.session_state["_enable_human_compare_next_run"] = True
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
                        st.session_state["_enable_human_compare_next_run"] = True
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
            human_sol_view = _get_cached_human_solution(problem, human_model)
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
                cache_key=(
                    "human_view",
                    prob_sig,
                    _history_signature(human_model),
                    _clamp_load_case_index(problem, selected_load_case),
                    log_strain,
                    False,
                ),
            )

        st.divider()
        st.subheader("🧠 LLM Designer Reasoning Transcript")
        llm_records = [
            sol
            for sol in loaded_solutions.values()
            if sol.solver_name == "llm_designer"
            and sol.metadata.get("transcript")
            and _problem_signature(sol.problem) == prob_sig
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
        default_export_idx = 0
        for idx, ek in enumerate(export_keys):
            if (
                ek != "Human Designer Canvas"
                and not ek.startswith("Replay: ")
                and _problem_signature(exportable_models[ek].problem) == prob_sig
            ):
                default_export_idx = idx
                break
        else:
            if "Human Designer Canvas" in exportable_models:
                default_export_idx = export_keys.index("Human Designer Canvas")

        if (
            "export_model_selector" in st.session_state
            and st.session_state["export_model_selector"] not in exportable_models
        ):
            st.session_state["export_model_selector"] = export_keys[default_export_idx]

        chosen_export_label = st.selectbox(
            "Select Parametric CADModel to Export",
            options=export_keys,
            index=default_export_idx,
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
