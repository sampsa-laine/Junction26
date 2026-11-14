from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .csg import CADModel, CutoutAction
from .fea import FEAResult, FEASolver
from .heuristic_agent import HeuristicCADDesigner
from .llm_designer import LLMCADDesigner
from .metrics import EvaluationScoreboard, evaluate_design
from .oracle_solver import ParametricCADOptimizer
from .problem import (
    HARBOR_TASK_BUILDERS,
    DesignProblem,
    make_cantilever_beam,
    make_l_bracket,
    make_mbb_beam,
    make_suspension_arm,
)
from .simp import SIMPOptimizer, SIMPResult
from .verifier import VerificationResult

SourceKind = Literal["harbor_run", "in_repo_solver", "design_file", "human"]
InRepoSolverName = Literal["simp", "oracle_solver", "heuristic_agent", "llm_designer"]


@dataclass(frozen=True)
class ProblemPreset:
    key: str
    display_name: str
    is_harbor_task: bool
    task_id: str | None
    builder: Callable[[], DesignProblem]


PROBLEM_PRESETS: dict[str, ProblemPreset] = {
    "cantilever-lightweight": ProblemPreset(
        key="cantilever-lightweight",
        display_name="Harbor 1: Cantilever Lightweight Bracket",
        is_harbor_task=True,
        task_id="cantilever-lightweight",
        builder=HARBOR_TASK_BUILDERS["cantilever-lightweight"],
    ),
    "l-bracket-stress": ProblemPreset(
        key="l-bracket-stress",
        display_name="Harbor 2: Stress-Constrained L-Bracket",
        is_harbor_task=True,
        task_id="l-bracket-stress",
        builder=HARBOR_TASK_BUILDERS["l-bracket-stress"],
    ),
    "suspension-rocker-multiload": ProblemPreset(
        key="suspension-rocker-multiload",
        display_name="Harbor 3: Multi-Load Suspension Rocker",
        is_harbor_task=True,
        task_id="suspension-rocker-multiload",
        builder=HARBOR_TASK_BUILDERS["suspension-rocker-multiload"],
    ),
    "aerospace-rib-obstacles": ProblemPreset(
        key="aerospace-rib-obstacles",
        display_name="Harbor 4: Aerospace Rib with Obstacle Tunnels",
        is_harbor_task=True,
        task_id="aerospace-rib-obstacles",
        builder=HARBOR_TASK_BUILDERS["aerospace-rib-obstacles"],
    ),
    "cantilever": ProblemPreset(
        key="cantilever",
        display_name="Arena: Cantilever Beam",
        is_harbor_task=False,
        task_id=None,
        builder=make_cantilever_beam,
    ),
    "mbb": ProblemPreset(
        key="mbb",
        display_name="Arena: MBB Beam",
        is_harbor_task=False,
        task_id=None,
        builder=make_mbb_beam,
    ),
    "l_bracket": ProblemPreset(
        key="l_bracket",
        display_name="Arena: L-Bracket",
        is_harbor_task=False,
        task_id=None,
        builder=make_l_bracket,
    ),
    "suspension": ProblemPreset(
        key="suspension",
        display_name="Arena: Suspension Arm",
        is_harbor_task=False,
        task_id=None,
        builder=make_suspension_arm,
    ),
}

_PROBLEM_NAME_TO_PRESET_KEY: dict[str, str] = {
    "cantilever lightweight bracket": "cantilever-lightweight",
    "cantilever-lightweight": "cantilever-lightweight",
    "stress-constrained l-bracket": "l-bracket-stress",
    "l-bracket-stress": "l-bracket-stress",
    "multi-load suspension rocker": "suspension-rocker-multiload",
    "suspension-rocker-multiload": "suspension-rocker-multiload",
    "aerospace rib with obstacle tunnels": "aerospace-rib-obstacles",
    "aerospace-rib-obstacles": "aerospace-rib-obstacles",
    "cantilever beam": "cantilever",
    "cantilever": "cantilever",
    "mbb beam": "mbb",
    "mbb": "mbb",
    "l-bracket": "l_bracket",
    "l_bracket": "l_bracket",
    "suspension arm": "suspension",
    "suspension": "suspension",
}

_SIMP_CACHE: dict[tuple[str, int], SIMPResult] = {}


@dataclass
class SolutionRecord:
    solution_id: str
    label: str
    source_kind: SourceKind
    solver_name: str
    task_id: str | None
    problem: DesignProblem
    cad_model: CADModel | None
    density: np.ndarray
    fea_result: FEAResult
    scoreboard: EvaluationScoreboard
    verification: VerificationResult | None
    harbor_reward: float | None
    metadata: dict[str, Any]


@dataclass
class ReplayFrame:
    step: int
    total_steps: int
    action: CutoutAction | None
    cad_model: CADModel
    density: np.ndarray
    fea_result: FEAResult
    volume_fraction: float
    compliance: float
    max_von_mises_stress: float
    structurally_connected: bool


def _problem_cache_key(problem: DesignProblem) -> str:
    return json.dumps(problem.to_dict(), sort_keys=True)


def _problem_short_id(problem: DesignProblem) -> str:
    return hashlib.sha256(_problem_cache_key(problem).encode("utf-8")).hexdigest()[:12]


def get_cached_reference_simp(
    problem: DesignProblem, max_iterations: int = 45
) -> SIMPResult:
    cache_key = (_problem_cache_key(problem), max_iterations)
    cached = _SIMP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = SIMPOptimizer(problem).optimize(max_iterations=max_iterations)
    _SIMP_CACHE[cache_key] = result
    return result


def infer_task_id_from_problem(problem: DesignProblem) -> str | None:
    preset_key = _PROBLEM_NAME_TO_PRESET_KEY.get(problem.name.strip().lower())
    if preset_key is None:
        return None
    return PROBLEM_PRESETS[preset_key].task_id


def _load_harbor_task_problem(task_id: str, tasks_root: Path) -> DesignProblem | None:
    task_problem_json = tasks_root / task_id / "environment" / "problem.json"
    if task_problem_json.is_file():
        return DesignProblem.load_json(task_problem_json)
    builder = HARBOR_TASK_BUILDERS.get(task_id)
    if builder is not None:
        return builder()
    return None


def _normalize_task_id_candidate(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    cleaned = raw_value.strip().rstrip("/")
    if "/" in cleaned:
        cleaned = cleaned.split("/")[-1]
    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[0]
    if cleaned in HARBOR_TASK_BUILDERS:
        return cleaned
    mapped = _PROBLEM_NAME_TO_PRESET_KEY.get(cleaned.lower())
    if mapped and PROBLEM_PRESETS[mapped].task_id:
        return PROBLEM_PRESETS[mapped].task_id
    return None


def _extract_task_id_from_trial_metadata(
    trial_dir: Path | None,
    result_payload: dict[str, Any] | None = None,
    config_payload: dict[str, Any] | None = None,
) -> str | None:
    candidates: list[str | None] = []
    if result_payload:
        candidates.append(_as_optional_str(result_payload.get("task_name")))
        candidates.append(_as_optional_str(result_payload.get("trial_name")))
        task_id_obj = result_payload.get("task_id")
        if isinstance(task_id_obj, dict):
            candidates.append(_as_optional_str(task_id_obj.get("path")))
            candidates.append(_as_optional_str(task_id_obj.get("name")))
        cfg_in_result = result_payload.get("config")
        if isinstance(cfg_in_result, dict):
            candidates.append(_as_optional_str(cfg_in_result.get("trial_name")))
            task_cfg = cfg_in_result.get("task")
            if isinstance(task_cfg, dict):
                candidates.append(_as_optional_str(task_cfg.get("name")))
                candidates.append(_as_optional_str(task_cfg.get("path")))

    if config_payload:
        candidates.append(_as_optional_str(config_payload.get("task_name")))
        candidates.append(_as_optional_str(config_payload.get("trial_name")))
        task_cfg = config_payload.get("task")
        if isinstance(task_cfg, dict):
            candidates.append(_as_optional_str(task_cfg.get("name")))
            candidates.append(_as_optional_str(task_cfg.get("path")))

    if trial_dir is not None:
        candidates.append(trial_dir.name)
        candidates.append(trial_dir.parent.name)

    for candidate in candidates:
        task_id = _normalize_task_id_candidate(candidate)
        if task_id is not None:
            return task_id
    return None


def resolve_problem_for_design(
    design_data: dict[str, Any] | None = None,
    problem: DesignProblem | None = None,
    search_dir: str | Path | None = None,
    result_payload: dict[str, Any] | None = None,
    config_payload: dict[str, Any] | None = None,
    tasks_root: str | Path = "tasks",
) -> tuple[DesignProblem, str | None]:
    if problem is not None:
        return problem, infer_task_id_from_problem(problem)

    tasks_dir = Path(tasks_root)
    base_dir = Path(search_dir) if search_dir is not None else None

    if base_dir is not None:
        for candidate_prob in (
            base_dir / "problem.json",
            base_dir / "environment" / "problem.json",
        ):
            if candidate_prob.is_file():
                loaded_prob = DesignProblem.load_json(candidate_prob)
                return loaded_prob, infer_task_id_from_problem(loaded_prob)

    task_id = _extract_task_id_from_trial_metadata(
        base_dir, result_payload=result_payload, config_payload=config_payload
    )
    if task_id is not None:
        task_prob = _load_harbor_task_problem(task_id, tasks_dir)
        if task_prob is not None:
            return task_prob, task_id

    problem_name = (
        _as_optional_str(design_data.get("problem_name")) if design_data else None
    )
    if problem_name:
        preset_key = _PROBLEM_NAME_TO_PRESET_KEY.get(problem_name.strip().lower())
        if preset_key is not None:
            preset = PROBLEM_PRESETS[preset_key]
            if preset.task_id is not None:
                task_prob = _load_harbor_task_problem(preset.task_id, tasks_dir)
                if task_prob is not None:
                    return task_prob, preset.task_id
            return preset.builder(), preset.task_id

    raise ValueError(
        "Unable to resolve DesignProblem from override, problem.json, trial metadata, or design problem_name."
    )


def evaluate_cad_verification(
    problem: DesignProblem,
    model: CADModel,
    reference_simp: SIMPResult | None = None,
    fea_result: FEAResult | None = None,
    simp_iterations: int = 45,
) -> VerificationResult:
    simp_res = reference_simp or get_cached_reference_simp(
        problem, max_iterations=simp_iterations
    )
    ref_c = float(simp_res.minimum_compliance)
    stock_vol = CADModel(problem).get_stock_volume_fraction()

    vol = model.get_volume_fraction()
    connected = model.is_structurally_connected()
    wall_ok, sliver_ratio = model.check_wall_thickness()
    violations = model.check_feature_constraints()

    if fea_result is None:
        density = model.to_density_grid()
        fea_result = FEASolver(problem).solve(density, penalty=1.0)

    comp = float(fea_result.compliance)
    max_vm = float(np.max(fea_result.von_mises_stress))

    if len(model.history) == 0 or vol >= stock_vol - 0.02:
        return VerificationResult(
            reward=0.0,
            valid=False,
            reason="Design is uncut or removes negligible material (<2% area).",
            volume_fraction=vol,
            target_volume_fraction=problem.target_volume_fraction,
            stock_volume_fraction=stock_vol,
            agent_compliance=comp,
            simp_reference_compliance=ref_c,
            case_compliances=list(fea_result.case_compliances),
            max_von_mises_stress=max_vm,
            max_stress_limit=problem.max_stress_limit,
            structurally_connected=connected,
            wall_thickness_ok=wall_ok,
            thin_sliver_ratio=sliver_ratio,
            stiffness_factor=0.0,
            volume_factor=0.0,
            stress_factor=0.0,
            manufacturability_factor=0.0,
            num_actions=len(model.history),
        )

    if violations:
        return VerificationResult(
            reward=0.0,
            valid=False,
            reason=f"CAD feature constraint violation: {'; '.join(violations)}",
            volume_fraction=vol,
            target_volume_fraction=problem.target_volume_fraction,
            stock_volume_fraction=stock_vol,
            agent_compliance=comp,
            simp_reference_compliance=ref_c,
            case_compliances=list(fea_result.case_compliances),
            max_von_mises_stress=max_vm,
            max_stress_limit=problem.max_stress_limit,
            structurally_connected=connected,
            wall_thickness_ok=wall_ok,
            thin_sliver_ratio=sliver_ratio,
            stiffness_factor=0.0,
            volume_factor=0.0,
            stress_factor=0.0,
            manufacturability_factor=0.0,
            num_actions=len(model.history),
        )

    if not connected or not math.isfinite(comp) or comp > ref_c * 20.0:
        return VerificationResult(
            reward=0.0,
            valid=False,
            reason="Load path severed or disconnected solid island detected.",
            volume_fraction=vol,
            target_volume_fraction=problem.target_volume_fraction,
            stock_volume_fraction=stock_vol,
            agent_compliance=comp,
            simp_reference_compliance=ref_c,
            case_compliances=list(fea_result.case_compliances),
            max_von_mises_stress=max_vm,
            max_stress_limit=problem.max_stress_limit,
            structurally_connected=connected,
            wall_thickness_ok=wall_ok,
            thin_sliver_ratio=sliver_ratio,
            stiffness_factor=0.0,
            volume_factor=0.0,
            stress_factor=0.0,
            manufacturability_factor=0.0,
            num_actions=len(model.history),
        )

    target_v = problem.target_volume_fraction
    vol_adjusted_comp = comp * max(1.0, vol / max(target_v, 1e-6))
    stiffness_factor = float(
        np.clip((ref_c / max(vol_adjusted_comp, 1e-9)) ** 0.75, 0.0, 1.0)
    )

    vol_tol = 0.005
    if vol <= target_v + vol_tol:
        volume_factor = 1.0
    else:
        excess = vol - (target_v + vol_tol)
        margin = max(0.12, (stock_vol - target_v) * 0.85)
        volume_factor = float(max(0.0, 1.0 - excess / margin) ** 2)

    if problem.max_stress_limit is not None and max_vm > problem.max_stress_limit:
        stress_factor = float(
            max(0.20, (problem.max_stress_limit / max(max_vm, 1e-9)) ** 1.5)
        )
    else:
        stress_factor = 1.0

    _, smoothness_factor = model.check_boundary_smoothness()
    if wall_ok:
        manufacturability_factor = float(smoothness_factor)
    else:
        sliver_pen = max(0.45, 1.0 - 3.0 * sliver_ratio)
        manufacturability_factor = float(0.40 * sliver_pen * smoothness_factor)

    raw_reward = (
        stiffness_factor * volume_factor * stress_factor * manufacturability_factor
    )
    reward = round(float(np.clip(raw_reward, 0.0, 1.0)), 4)

    return VerificationResult(
        reward=reward,
        valid=True,
        reason="Valid parametric CAD design evaluated.",
        volume_fraction=round(vol, 4),
        target_volume_fraction=round(target_v, 4),
        stock_volume_fraction=round(stock_vol, 4),
        agent_compliance=round(comp, 4),
        simp_reference_compliance=round(ref_c, 4),
        case_compliances=[round(c, 4) for c in fea_result.case_compliances],
        max_von_mises_stress=round(max_vm, 4),
        max_stress_limit=problem.max_stress_limit,
        structurally_connected=connected,
        wall_thickness_ok=wall_ok,
        thin_sliver_ratio=round(sliver_ratio, 4),
        stiffness_factor=round(stiffness_factor, 4),
        volume_factor=round(volume_factor, 4),
        stress_factor=round(stress_factor, 4),
        manufacturability_factor=round(manufacturability_factor, 4),
        num_actions=len(model.history),
    )


def _as_optional_str(val: Any) -> str | None:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _read_optional_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text())
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _is_contained_file(path: Path, root: Path) -> bool:
    try:
        return path.is_file() and path.resolve().is_relative_to(root)
    except OSError:
        return False


def _is_contained_dir(path: Path, root: Path) -> bool:
    try:
        return path.is_dir() and path.resolve().is_relative_to(root)
    except OSError:
        return False


def _locate_design_json(trial_dir: Path) -> Path | None:
    try:
        if not trial_dir.is_dir():
            return None
        trial_root = trial_dir.resolve()
    except OSError:
        return None

    artifacts_dir = trial_dir / "artifacts"
    manifest_path = artifacts_dir / "manifest.json"
    if _is_contained_file(manifest_path, trial_root):
        try:
            manifest = json.loads(manifest_path.read_text())
            destinations: list[str] = []
            if isinstance(manifest, list):
                for entry in manifest:
                    if not isinstance(entry, dict):
                        continue
                    src = str(entry.get("source", ""))
                    dst = str(entry.get("destination", ""))
                    if src.endswith("design.json") or dst.endswith("design.json"):
                        if dst:
                            destinations.append(dst)
            elif isinstance(manifest, dict):
                for src, dst in manifest.items():
                    if str(src).endswith("design.json") or str(dst).endswith(
                        "design.json"
                    ):
                        destinations.append(str(dst))
            for dst in destinations:
                rel_dst = dst.lstrip("/")
                if not rel_dst:
                    continue
                for candidate in (artifacts_dir / rel_dst, trial_dir / rel_dst):
                    if _is_contained_file(candidate, trial_root):
                        return candidate
        except Exception:
            pass

    for rel_path in (
        Path("artifacts") / "app" / "design.json",
        Path("artifacts") / "design.json",
        Path("design.json"),
    ):
        candidate = trial_dir / rel_path
        if _is_contained_file(candidate, trial_root):
            return candidate

    if _is_contained_dir(artifacts_dir, trial_root):
        found: list[Path] = []
        visited_dirs: set[Path] = set()

        def _walk_artifacts(curr_dir: Path) -> None:
            try:
                curr_resolved = curr_dir.resolve()
            except OSError:
                return
            if curr_resolved in visited_dirs or not curr_resolved.is_relative_to(
                trial_root
            ):
                return
            visited_dirs.add(curr_resolved)
            try:
                entries = sorted(curr_dir.iterdir())
            except OSError:
                return
            for entry in entries:
                if entry.name == "design.json" and _is_contained_file(
                    entry, trial_root
                ):
                    found.append(entry)
                elif _is_contained_dir(entry, trial_root):
                    _walk_artifacts(entry)

        _walk_artifacts(artifacts_dir)
        if found:
            return sorted(found)[0]

    return None


def _infer_enclosing_trial_dir(design_file: Path) -> Path:
    for ancestor in design_file.parents:
        if ancestor.name == "artifacts" and ancestor.parent != ancestor:
            return ancestor.parent
    return design_file.parent


def discover_harbor_runs(root: str | Path = "jobs") -> list[Path]:
    root_path = Path(root)
    try:
        if not root_path.exists():
            return []
        if root_path.is_file():
            return [root_path] if root_path.name.endswith(".json") else []
        root_resolved = root_path.resolve()
    except OSError:
        return []

    discovered: list[Path] = []
    seen_trials: set[Path] = set()
    visited_dirs: set[Path] = set()
    skipped_subdir_names = {"artifacts", "verifier", "agent", "logs", "__pycache__"}

    def _walk(current: Path) -> None:
        try:
            curr_resolved = current.resolve()
        except OSError:
            return
        if curr_resolved in visited_dirs or not curr_resolved.is_relative_to(
            root_resolved
        ):
            return
        visited_dirs.add(curr_resolved)

        if _locate_design_json(current) is not None:
            if curr_resolved not in seen_trials:
                seen_trials.add(curr_resolved)
                discovered.append(current)
            return

        try:
            entries = list(current.iterdir())
        except OSError:
            return

        subdirs: list[tuple[bool, str, Path]] = []
        for entry in entries:
            if entry.name in skipped_subdir_names or entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    subdirs.append((entry.is_symlink(), entry.name, entry))
            except OSError:
                continue

        for _, _, child in sorted(subdirs, key=lambda item: (item[0], item[1])):
            _walk(child)

    _walk(root_path)
    return discovered


def _parse_verification_metrics(metrics_path: Path) -> VerificationResult | None:
    payload = _read_optional_json_dict(metrics_path)
    if payload is None:
        return None
    try:
        raw_comp = float(payload["agent_compliance"])
        agent_comp = float("inf") if raw_comp < 0.0 else raw_comp
        max_limit_raw = payload.get("max_stress_limit")
        max_limit = float(max_limit_raw) if max_limit_raw is not None else None
        return VerificationResult(
            reward=float(payload["reward"]),
            valid=bool(payload["valid"]),
            reason=str(payload["reason"]),
            volume_fraction=float(payload["volume_fraction"]),
            target_volume_fraction=float(payload["target_volume_fraction"]),
            stock_volume_fraction=float(payload["stock_volume_fraction"]),
            agent_compliance=agent_comp,
            simp_reference_compliance=float(payload["simp_reference_compliance"]),
            case_compliances=[float(v) for v in payload.get("case_compliances", [])],
            max_von_mises_stress=float(payload["max_von_mises_stress"]),
            max_stress_limit=max_limit,
            structurally_connected=bool(payload["structurally_connected"]),
            wall_thickness_ok=bool(payload["wall_thickness_ok"]),
            thin_sliver_ratio=float(payload["thin_sliver_ratio"]),
            stiffness_factor=float(payload["stiffness_factor"]),
            volume_factor=float(payload["volume_factor"]),
            stress_factor=float(payload["stress_factor"]),
            manufacturability_factor=float(payload["manufacturability_factor"]),
            num_actions=int(payload["num_actions"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _read_optional_reward(
    trial_dir: Path, result_payload: dict[str, Any] | None
) -> float | None:
    for rel in (
        Path("verifier") / "reward.txt",
        Path("logs") / "verifier" / "reward.txt",
        Path("reward.txt"),
    ):
        reward_file = trial_dir / rel
        if reward_file.is_file():
            try:
                return float(reward_file.read_text().strip().splitlines()[0])
            except (IndexError, ValueError):
                pass

    if result_payload:
        verifier_res = result_payload.get("verifier_result")
        if isinstance(verifier_res, dict):
            rewards = verifier_res.get("rewards")
            if isinstance(rewards, dict) and rewards.get("reward") is not None:
                try:
                    return float(rewards["reward"])
                except (TypeError, ValueError):
                    pass
    return None


def _extract_agent_solver_name(
    result_payload: dict[str, Any] | None,
    config_payload: dict[str, Any] | None,
) -> str:
    agent_name: str | None = None
    model_name: str | None = None

    for container in (result_payload, config_payload):
        if not container:
            continue
        cfg = container.get("config") if "config" in container else container
        if isinstance(cfg, dict):
            agent_cfg = cfg.get("agent")
            if isinstance(agent_cfg, dict):
                agent_name = agent_name or _as_optional_str(agent_cfg.get("name"))
                model_name = model_name or _as_optional_str(agent_cfg.get("model_name"))
        agent_info = container.get("agent_info")
        if isinstance(agent_info, dict):
            agent_name = agent_name or _as_optional_str(agent_info.get("name"))
            model_info = agent_info.get("model_info")
            if isinstance(model_info, str):
                model_name = model_name or _as_optional_str(model_info)
            elif isinstance(model_info, dict):
                model_name = model_name or _as_optional_str(model_info.get("name"))

    if agent_name and model_name:
        return f"{agent_name} ({model_name})"
    return agent_name or "harbor_agent"


def load_harbor_trial(
    trial_or_artifact_path: str | Path,
    problem: DesignProblem | None = None,
    tasks_root: str | Path = "tasks",
    reference_simp: SIMPResult | None = None,
) -> SolutionRecord:
    input_path = Path(trial_or_artifact_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Harbor path does not exist: {input_path}")

    if input_path.is_file():
        design_path = input_path
        trial_dir = _infer_enclosing_trial_dir(design_path)
    else:
        located = _locate_design_json(input_path)
        if located is not None:
            trial_dir = input_path
            design_path = located
        else:
            nested = discover_harbor_runs(input_path)
            if len(nested) == 1:
                trial_dir = nested[0]
                located_nested = _locate_design_json(trial_dir)
                if located_nested is None:
                    raise FileNotFoundError(
                        f"Could not locate design.json in {trial_dir}"
                    )
                design_path = located_nested
            else:
                raise FileNotFoundError(
                    f"Could not locate design.json in Harbor trial directory: {input_path}"
                )

    design_data = json.loads(design_path.read_text())
    if not isinstance(design_data, dict):
        raise ValueError(
            f"Expected JSON object in {design_path}, got {type(design_data).__name__}."
        )
    result_payload = _read_optional_json_dict(trial_dir / "result.json")
    config_payload = _read_optional_json_dict(trial_dir / "config.json")

    resolved_problem, task_id = resolve_problem_for_design(
        design_data=design_data,
        problem=problem,
        search_dir=trial_dir,
        result_payload=result_payload,
        config_payload=config_payload,
        tasks_root=tasks_root,
    )

    cad_model = CADModel.from_dict(design_data, resolved_problem)
    solver = FEASolver(resolved_problem)
    density = cad_model.to_density_grid()
    fea_res = solver.solve(density, penalty=1.0)

    ref_simp = reference_simp or get_cached_reference_simp(resolved_problem)
    solver_label = _extract_agent_solver_name(result_payload, config_payload)
    trial_name = (
        _as_optional_str(result_payload.get("trial_name")) if result_payload else None
    ) or trial_dir.name
    display_label = f"Harbor [{solver_label}] — {trial_name}"

    scoreboard = evaluate_design(
        display_label, density, resolved_problem, reference_simp=ref_simp, solver=solver
    )

    verification: VerificationResult | None = None
    for rel_metrics in (
        Path("verifier") / "metrics.json",
        Path("logs") / "verifier" / "metrics.json",
        Path("metrics.json"),
    ):
        cand_metrics = trial_dir / rel_metrics
        if cand_metrics.is_file():
            verification = _parse_verification_metrics(cand_metrics)
            if verification is not None:
                break

    if verification is None:
        verification = evaluate_cad_verification(
            resolved_problem,
            cad_model,
            reference_simp=ref_simp,
            fea_result=fea_res,
        )

    reward_val = _read_optional_reward(trial_dir, result_payload)
    if reward_val is None:
        reward_val = verification.reward

    return SolutionRecord(
        solution_id=f"harbor:{trial_dir.resolve()}",
        label=display_label,
        source_kind="harbor_run",
        solver_name=solver_label,
        task_id=task_id,
        problem=resolved_problem,
        cad_model=cad_model,
        density=density,
        fea_result=fea_res,
        scoreboard=scoreboard,
        verification=verification,
        harbor_reward=reward_val,
        metadata={
            "trial_dir": str(trial_dir),
            "design_path": str(design_path),
            "trial_name": trial_name,
            "started_at": result_payload.get("started_at") if result_payload else None,
            "finished_at": result_payload.get("finished_at")
            if result_payload
            else None,
        },
    )


def load_design_payload_solution(
    design_data: dict[str, Any],
    problem: DesignProblem | None = None,
    label: str | None = None,
    tasks_root: str | Path = "tasks",
    reference_simp: SIMPResult | None = None,
    search_dir: str | Path | None = None,
    source_path: str | None = None,
) -> SolutionRecord:
    if not isinstance(design_data, dict):
        raise ValueError(
            f"Expected design payload dict, got {type(design_data).__name__}."
        )
    resolved_problem, task_id = resolve_problem_for_design(
        design_data=design_data,
        problem=problem,
        search_dir=search_dir,
        tasks_root=tasks_root,
    )
    cad_model = CADModel.from_dict(design_data, resolved_problem)
    solver = FEASolver(resolved_problem)
    density = cad_model.to_density_grid()
    fea_res = solver.solve(density, penalty=1.0)
    ref_simp = reference_simp or get_cached_reference_simp(resolved_problem)

    display_label = label or f"Design JSON ({resolved_problem.name})"
    scoreboard = evaluate_design(
        display_label, density, resolved_problem, reference_simp=ref_simp, solver=solver
    )
    verification = evaluate_cad_verification(
        resolved_problem,
        cad_model,
        reference_simp=ref_simp,
        fea_result=fea_res,
    )
    sol_key = source_path or display_label
    return SolutionRecord(
        solution_id=f"design_file:{sol_key}",
        label=display_label,
        source_kind="design_file",
        solver_name="design_json",
        task_id=task_id,
        problem=resolved_problem,
        cad_model=cad_model,
        density=density,
        fea_result=fea_res,
        scoreboard=scoreboard,
        verification=verification,
        harbor_reward=verification.reward,
        metadata={"design_path": source_path},
    )


def load_design_solution(
    design_path: str | Path,
    problem: DesignProblem | None = None,
    label: str | None = None,
    tasks_root: str | Path = "tasks",
    reference_simp: SIMPResult | None = None,
) -> SolutionRecord:
    d_path = Path(design_path)
    if not d_path.is_file():
        raise FileNotFoundError(f"Design JSON file not found: {d_path}")
    design_data = json.loads(d_path.read_text())
    if not isinstance(design_data, dict):
        raise ValueError(
            f"Expected JSON object in {d_path}, got {type(design_data).__name__}."
        )
    resolved_label = label or f"Design File ({d_path.name})"
    return load_design_payload_solution(
        design_data=design_data,
        problem=problem,
        label=resolved_label,
        tasks_root=tasks_root,
        reference_simp=reference_simp,
        search_dir=d_path.parent,
        source_path=str(d_path.resolve()),
    )


load_design_json_solution = load_design_solution


def run_in_repo_solver(
    problem: DesignProblem,
    solver_name: InRepoSolverName,
    reference_simp: SIMPResult | None = None,
    **kwargs: Any,
) -> SolutionRecord:
    normalized = solver_name.strip().lower()
    task_id = infer_task_id_from_problem(problem)
    solver = FEASolver(problem)
    simp_iters = int(kwargs.pop("simp_iterations", 45))

    if normalized == "simp":
        max_iterations = int(kwargs.pop("max_iterations", simp_iters))
        simp_res = SIMPOptimizer(problem).optimize(
            max_iterations=max_iterations, **kwargs
        )
        _SIMP_CACHE[(_problem_cache_key(problem), max_iterations)] = simp_res
        density = simp_res.optimal_density.copy()
        fea_res = solver.solve(density, penalty=1.0)
        label = "SIMP Topology Optimum (Continuous)"
        scoreboard = evaluate_design(
            label, density, problem, reference_simp=simp_res, solver=solver
        )
        return SolutionRecord(
            solution_id=f"solver:simp:{_problem_short_id(problem)}",
            label=label,
            source_kind="in_repo_solver",
            solver_name="simp",
            task_id=task_id,
            problem=problem,
            cad_model=None,
            density=density,
            fea_result=fea_res,
            scoreboard=scoreboard,
            verification=None,
            harbor_reward=None,
            metadata={
                "iterations": simp_res.iterations,
                "compliance_history": list(simp_res.compliance_history),
            },
        )

    ref_simp = reference_simp or get_cached_reference_simp(
        problem, max_iterations=simp_iters
    )
    extra_metadata: dict[str, Any] = {}

    if normalized == "oracle_solver":
        cad_model = ParametricCADOptimizer(problem).optimize()
        label = "Oracle Solver (SIMP-to-CAD)"
        canonical_solver = "oracle_solver"
    elif normalized == "heuristic_agent":
        cad_model = HeuristicCADDesigner(problem).design(**kwargs)
        label = "Heuristic CAD Agent"
        canonical_solver = "heuristic_agent"
    elif normalized == "llm_designer":
        llm_res = LLMCADDesigner(problem).design(**kwargs)
        cad_model = llm_res.cad_model
        label = "LLM ReAct Designer"
        canonical_solver = "llm_designer"
        extra_metadata["transcript"] = llm_res.transcript
    else:
        raise ValueError(
            f"Unsupported solver_name {solver_name!r}; expected one of "
            "'simp', 'oracle_solver', 'heuristic_agent', 'llm_designer'."
        )

    density = cad_model.to_density_grid()
    fea_res = solver.solve(density, penalty=1.0)
    scoreboard = evaluate_design(
        label, density, problem, reference_simp=ref_simp, solver=solver
    )
    verification = evaluate_cad_verification(
        problem, cad_model, reference_simp=ref_simp, fea_result=fea_res
    )
    return SolutionRecord(
        solution_id=f"solver:{canonical_solver}:{_problem_short_id(problem)}",
        label=label,
        source_kind="in_repo_solver",
        solver_name=canonical_solver,
        task_id=task_id,
        problem=problem,
        cad_model=cad_model,
        density=density,
        fea_result=fea_res,
        scoreboard=scoreboard,
        verification=verification,
        harbor_reward=verification.reward,
        metadata=extra_metadata,
    )


def build_human_solution(
    problem: DesignProblem,
    cad_model: CADModel,
    label: str = "Human Designer",
    reference_simp: SIMPResult | None = None,
) -> SolutionRecord:
    solver = FEASolver(problem)
    density = cad_model.to_density_grid()
    fea_res = solver.solve(density, penalty=1.0)
    ref_simp = reference_simp or get_cached_reference_simp(problem)
    scoreboard = evaluate_design(
        label, density, problem, reference_simp=ref_simp, solver=solver
    )
    verification = evaluate_cad_verification(
        problem, cad_model, reference_simp=ref_simp, fea_result=fea_res
    )
    return SolutionRecord(
        solution_id=f"human:{_problem_short_id(problem)}",
        label=label,
        source_kind="human",
        solver_name="human",
        task_id=infer_task_id_from_problem(problem),
        problem=problem,
        cad_model=cad_model,
        density=density,
        fea_result=fea_res,
        scoreboard=scoreboard,
        verification=verification,
        harbor_reward=verification.reward,
        metadata={},
    )


def build_comparison_rows(solutions: Sequence[SolutionRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sol in solutions:
        ver = sol.verification
        stress_limit = sol.problem.max_stress_limit
        max_vm = float(np.max(sol.fea_result.von_mises_stress))
        stress_str = (
            f"{max_vm:.3f} / {stress_limit:.2f}"
            if stress_limit is not None
            else f"{max_vm:.3f}"
        )
        case_comp_str = ", ".join(f"{c:.2f}" for c in sol.fea_result.case_compliances)
        comp_val = sol.fea_result.compliance
        comp_str = f"{comp_val:.2f}" if math.isfinite(comp_val) else "inf"

        rows.append(
            {
                "Solution": sol.label,
                "Source": sol.source_kind,
                "Solver": sol.solver_name,
                "Problem": sol.problem.name,
                "Harbor Reward": (
                    f"{sol.harbor_reward:.4f}"
                    if sol.harbor_reward is not None
                    else "N/A (Non-CAD)"
                ),
                "Arena Score": f"{sol.scoreboard.overall_score:.1f}",
                "Compliance": comp_str,
                "Case Compliances": case_comp_str,
                "SIMP Gap %": (
                    "0.0%"
                    if sol.solver_name == "simp"
                    else f"+{sol.scoreboard.optimality_gap_percent:.1f}%"
                ),
                "Volume / Target": (
                    f"{sol.scoreboard.volume_fraction:.3f} / "
                    f"{sol.problem.target_volume_fraction:.2f}"
                ),
                "Max Stress / Limit": stress_str,
                "Connected": (
                    "Yes"
                    if ver is not None and ver.structurally_connected
                    else ("No" if ver is not None else "N/A")
                ),
                "Wall OK": (
                    "Yes"
                    if ver is not None and ver.wall_thickness_ok
                    else ("No" if ver is not None else "N/A")
                ),
                "Cutouts": len(sol.cad_model.history)
                if sol.cad_model is not None
                else 0,
            }
        )
    return rows


def reconstruct_cad_step(model: CADModel, step: int) -> CADModel:
    total_steps = len(model.history)
    if step < 0 or step > total_steps:
        raise ValueError(
            f"Replay step {step} is out of bounds for history length {total_steps}."
        )
    replayed = CADModel(model.problem)
    if step > 0:
        replayed.history = [replace(a) for a in model.history[:step]]
        replayed._rebuild_geometry()
    return replayed


def _solve_for_load_case(
    problem: DesignProblem,
    density: np.ndarray,
    solver: FEASolver | None = None,
    load_case_index: int | None = None,
) -> FEAResult:
    if load_case_index is None:
        active_solver = solver or FEASolver(problem)
        return active_solver.solve(density, penalty=1.0)

    cases = problem.effective_load_cases
    if load_case_index < 0 or load_case_index >= len(cases):
        raise ValueError(
            f"load_case_index {load_case_index} out of bounds for {len(cases)} load cases."
        )
    selected_case = list(cases[load_case_index])
    case_problem = replace(problem, loads=selected_case, load_cases=[selected_case])
    return FEASolver(case_problem).solve(density, penalty=1.0)


def evaluate_replay_step(
    model: CADModel,
    step: int,
    solver: FEASolver | None = None,
    load_case_index: int | None = None,
) -> ReplayFrame:
    step_model = reconstruct_cad_step(model, step)
    density = step_model.to_density_grid()
    fea_res = _solve_for_load_case(
        model.problem, density, solver=solver, load_case_index=load_case_index
    )
    action = replace(model.history[step - 1]) if step > 0 else None
    return ReplayFrame(
        step=step,
        total_steps=len(model.history),
        action=action,
        cad_model=step_model,
        density=density,
        fea_result=fea_res,
        volume_fraction=step_model.get_volume_fraction(),
        compliance=float(fea_res.compliance),
        max_von_mises_stress=float(np.max(fea_res.von_mises_stress)),
        structurally_connected=step_model.is_structurally_connected(),
    )


def replay_cutout_history(
    model: CADModel,
    solver: FEASolver | None = None,
    load_case_index: int | None = None,
) -> list[ReplayFrame]:
    shared_solver = (
        solver
        if solver is not None
        else (FEASolver(model.problem) if load_case_index is None else None)
    )
    return [
        evaluate_replay_step(
            model,
            step=step_idx,
            solver=shared_solver,
            load_case_index=load_case_index,
        )
        for step_idx in range(len(model.history) + 1)
    ]
