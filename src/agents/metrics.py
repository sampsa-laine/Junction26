from dataclasses import dataclass
import numpy as np

from .csg import CADModel
from .fea import FEASolver
from .problem import DesignProblem
from .simp import SIMPResult


@dataclass
class EvaluationScoreboard:
    designer_name: str
    compliance: float
    reference_compliance: float
    optimality_gap_percent: float
    max_von_mises_stress: float
    volume_fraction: float
    target_volume_fraction: float
    budget_satisfied: bool
    geometric_iou: float
    overall_score: float  # Normalized 0-100 composite score


def evaluate_design(
    name: str,
    density_field: np.ndarray,
    problem: DesignProblem,
    reference_simp: SIMPResult | None = None,
    solver: FEASolver | None = None,
) -> EvaluationScoreboard:
    """Evaluates a component design against FEA structural physics and reference SIMP TopOpt."""
    if solver is None:
        solver = FEASolver(problem)

    fea_result = solver.solve(density_field, penalty=1.0)
    current_vol = float(np.mean(density_field))
    target_vol = problem.target_volume_fraction
    budget_met = current_vol <= target_vol + 0.02

    ref_comp = (
        reference_simp.minimum_compliance if reference_simp else fea_result.compliance
    )

    # Optimality gap: how much higher compliance is compared to SIMP lower bound
    gap = max(0.0, (fea_result.compliance - ref_comp) / (ref_comp + 1e-12) * 100.0)

    # Geometric IoU with SIMP thresholded shape
    if reference_simp is not None:
        b_current = density_field > 0.5
        b_ref = reference_simp.optimal_density > 0.5
        intersection = np.logical_and(b_current, b_ref).sum()
        union = np.logical_or(b_current, b_ref).sum()
        iou = float(intersection / (union + 1e-12))
    else:
        iou = 1.0

    # Overall composite score (0 to 100)
    # Higher score = stiffer, closer to SIMP, meets volume constraint
    stiffness_score = max(0.0, 100.0 - gap)
    penalty_excess_vol = (
        max(0.0, current_vol - target_vol) * 200.0 if not budget_met else 0.0
    )
    overall = max(0.0, min(100.0, stiffness_score - penalty_excess_vol))

    return EvaluationScoreboard(
        designer_name=name,
        compliance=fea_result.compliance,
        reference_compliance=ref_comp,
        optimality_gap_percent=gap,
        max_von_mises_stress=float(np.max(fea_result.von_mises_stress)),
        volume_fraction=current_vol,
        target_volume_fraction=target_vol,
        budget_satisfied=budget_met,
        geometric_iou=iou,
        overall_score=overall,
    )


def evaluate_cad_model(
    name: str,
    cad_model: CADModel,
    problem: DesignProblem,
    reference_simp: SIMPResult | None = None,
    solver: FEASolver | None = None,
) -> EvaluationScoreboard:
    """Evaluates a parametric CSG CAD model."""
    density = cad_model.to_density_grid()
    return evaluate_design(name, density, problem, reference_simp, solver)
