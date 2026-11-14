import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .csg import CADModel
from .fea import FEASolver
from .problem import DesignProblem
from .simp import SIMPOptimizer


@dataclass
class VerificationResult:
    reward: float
    valid: bool
    reason: str
    volume_fraction: float
    target_volume_fraction: float
    stock_volume_fraction: float
    agent_compliance: float
    simp_reference_compliance: float
    case_compliances: list[float]
    max_von_mises_stress: float
    max_stress_limit: float | None
    structurally_connected: bool
    wall_thickness_ok: bool
    thin_sliver_ratio: float
    stiffness_factor: float
    volume_factor: float
    stress_factor: float
    manufacturability_factor: float
    num_actions: int


def score_cad_submission(
    problem: DesignProblem,
    design_path: str | Path,
    simp_iterations: int = 45,
) -> VerificationResult:
    """Computes continuous [0.0, 1.0] reward comparing parametric CAD design against SIMP C*."""
    d_path = Path(design_path)
    simp_res = SIMPOptimizer(problem).optimize(max_iterations=simp_iterations)
    ref_c = float(simp_res.minimum_compliance)

    stock_model = CADModel(problem)
    stock_vol = stock_model.get_stock_volume_fraction()

    if not d_path.exists():
        return VerificationResult(
            reward=0.0,
            valid=False,
            reason=f"Design file not found at {d_path}",
            volume_fraction=stock_vol,
            target_volume_fraction=problem.target_volume_fraction,
            stock_volume_fraction=stock_vol,
            agent_compliance=float("inf"),
            simp_reference_compliance=ref_c,
            case_compliances=[],
            max_von_mises_stress=0.0,
            max_stress_limit=problem.max_stress_limit,
            structurally_connected=False,
            wall_thickness_ok=False,
            thin_sliver_ratio=0.0,
            stiffness_factor=0.0,
            volume_factor=0.0,
            stress_factor=0.0,
            manufacturability_factor=0.0,
            num_actions=0,
        )

    try:
        model = CADModel.load_json(d_path, problem)
    except Exception as exc:
        return VerificationResult(
            reward=0.0,
            valid=False,
            reason=f"Invalid design JSON: {exc}",
            volume_fraction=stock_vol,
            target_volume_fraction=problem.target_volume_fraction,
            stock_volume_fraction=stock_vol,
            agent_compliance=float("inf"),
            simp_reference_compliance=ref_c,
            case_compliances=[],
            max_von_mises_stress=0.0,
            max_stress_limit=problem.max_stress_limit,
            structurally_connected=False,
            wall_thickness_ok=False,
            thin_sliver_ratio=0.0,
            stiffness_factor=0.0,
            volume_factor=0.0,
            stress_factor=0.0,
            manufacturability_factor=0.0,
            num_actions=0,
        )

    vol = model.get_volume_fraction()
    connected = model.is_structurally_connected()
    wall_ok, sliver_ratio = model.check_wall_thickness()
    violations = model.check_feature_constraints()

    solver = FEASolver(problem)
    density = model.to_density_grid()
    fea_res = solver.solve(density, penalty=1.0)
    comp = float(fea_res.compliance)
    max_vm = float(np.max(fea_res.von_mises_stress))

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
            case_compliances=fea_res.case_compliances,
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
            case_compliances=fea_res.case_compliances,
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
            case_compliances=fea_res.case_compliances,
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
        case_compliances=[round(c, 4) for c in fea_res.case_compliances],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harbor CAD TopOpt Reward Verifier")
    parser.add_argument("--problem", default="/app/problem.json")
    parser.add_argument("--design", default="/app/design.json")
    parser.add_argument("--reward-out", default="/logs/verifier/reward.txt")
    parser.add_argument("--metrics-out", default="/logs/verifier/metrics.json")
    args = parser.parse_args(argv)

    reward_path = Path(args.reward_out)
    reward_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        problem = DesignProblem.load_json(args.problem)
        res = score_cad_submission(problem, args.design)
    except Exception as exc:
        reward_path.write_text("0.0\n")
        print(f"Verifier error: {exc}")
        return 0

    reward_path.write_text(f"{res.reward:.4f}\n")
    if args.metrics_out:
        m_path = Path(args.metrics_out)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(res)
        if not math.isfinite(payload["agent_compliance"]):
            payload["agent_compliance"] = -1.0
        m_path.write_text(json.dumps(payload, indent=2))

    print(f"=== HARBOR VERIFIER REPORT ({problem.name}) ===")
    print(f"Status          : {'VALID' if res.valid else 'INVALID'} ({res.reason})")
    print(
        f"Volume Fraction : {res.volume_fraction:.4f} / target {res.target_volume_fraction:.4f}"
    )
    print(
        f"Agent Compliance: {res.agent_compliance:.4f} vs SIMP C*: {res.simp_reference_compliance:.4f}"
    )
    print(
        f"Max von Mises   : {res.max_von_mises_stress:.4f} (limit: {res.max_stress_limit})"
    )
    print(
        f"Factors         : stiff={res.stiffness_factor:.3f}, vol={res.volume_factor:.3f}, "
        f"stress={res.stress_factor:.3f}, mfg={res.manufacturability_factor:.3f}"
    )
    print(f"FINAL REWARD    : {res.reward:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
