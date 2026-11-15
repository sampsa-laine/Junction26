from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .csg import CADModel, CutoutAction
from .fea import FEAResult, FEASolver
from .problem import DesignProblem, KeepOutType
from .solutions import ReplayFrame, SolutionRecord

FieldKind = Literal["stress", "strain_energy"]

_BG_FIG = "#0f172a"
_BG_PLOT = "#0b1120"
_BG_CARD = "#162033"
_BORDER = "#1e293b"
_TEXT_PRIMARY = "#f8fafc"
_TEXT_MUTED = "#94a3b8"
_MODEL_FILL = "#8b9bb4"
_MODEL_EDGE = "#cbd5e1"
_MODEL_CMAP = LinearSegmentedColormap.from_list(
    "metallic_steel_model", [_BG_PLOT, "#475569", _MODEL_FILL, _MODEL_EDGE]
)
_LOAD_CASE_COLORS = ("#ff2a85", "#00f5ff", "#ffe600", "#d946ef")


@dataclass
class InspectionViews:
    model_figure: Figure
    stress_figure: Figure
    strain_figure: Figure
    fea_result: FEAResult
    density: np.ndarray
    load_case_index: int | None


def _new_figure(figsize: tuple[float, float] = (5.0, 3.2), dpi: int = 110) -> Figure:
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=_BG_FIG)
    FigureCanvasAgg(fig)
    return fig


def _style_dark_axis(ax: Axes) -> None:
    ax.set_facecolor(_BG_PLOT)
    ax.tick_params(colors="#ffffff", labelsize=7)
    ax.xaxis.label.set_color(_TEXT_MUTED)
    ax.yaxis.label.set_color(_TEXT_MUTED)
    ax.title.set_color(_TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_color(_BORDER)


def _polygon_to_compound_path(poly: Polygon) -> MplPath | None:
    oriented = orient(poly, sign=1.0)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []

    for ring in [oriented.exterior, *oriented.interiors]:
        coords = list(ring.coords)
        if len(coords) < 3:
            continue
        vertices.append((float(coords[0][0]), float(coords[0][1])))
        codes.append(int(MplPath.MOVETO))
        for x, y in coords[1:-1]:
            vertices.append((float(x), float(y)))
            codes.append(int(MplPath.LINETO))
        vertices.append((float(coords[-1][0]), float(coords[-1][1])))
        codes.append(int(MplPath.CLOSEPOLY))

    if not vertices:
        return None
    return MplPath(np.asarray(vertices, dtype=float), codes)


def compute_load_case_fea(
    problem: DesignProblem,
    density: np.ndarray,
    load_case_index: int | None = None,
    penalty: float = 1.0,
) -> FEAResult:
    if load_case_index is None:
        return FEASolver(problem).solve(density, penalty=penalty)

    cases = problem.effective_load_cases
    if load_case_index < 0 or load_case_index >= len(cases):
        raise ValueError(
            f"load_case_index {load_case_index} out of range for {len(cases)} load cases."
        )
    selected_loads = list(cases[load_case_index])
    single_case_problem = replace(
        problem, loads=selected_loads, load_cases=[selected_loads]
    )
    return FEASolver(single_case_problem).solve(density, penalty=penalty)


def _draw_keepouts(ax: Axes, problem: DesignProblem) -> None:
    for ko in problem.keepouts:
        is_solid = ko.keepout_type == KeepOutType.SOLID
        edge = "#c084fc" if is_solid else "#64748b"
        face = (0.75, 0.52, 0.99, 0.16) if is_solid else (0.04, 0.05, 0.09, 0.85)
        style = ":" if is_solid else "--"
        if ko.shape == "circle":
            cx, cy, r = ko.params
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r,
                    facecolor=face,
                    edgecolor=edge,
                    linestyle=style,
                    linewidth=1.1,
                    zorder=3,
                )
            )
        elif ko.shape == "rect":
            rx, ry, rw, rh = ko.params
            ax.add_patch(
                Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor=face,
                    edgecolor=edge,
                    linestyle=style,
                    linewidth=1.1,
                    zorder=3,
                )
            )


def _support_bar_geometries(problem: DesignProblem, rad: float) -> list[Polygon]:
    """Builds unified capsule/stadium shapes for collinear support points."""
    supports = problem.supports
    if not supports:
        return []
    used: set[int] = set()
    shapes = []
    for i, s in enumerate(supports):
        if i in used:
            continue
        same_x = [
            j for j, o in enumerate(supports) if j not in used and abs(o.x - s.x) < 1e-3
        ]
        same_y = [
            j for j, o in enumerate(supports) if j not in used and abs(o.y - s.y) < 1e-3
        ]
        group = same_x if len(same_x) >= len(same_y) else same_y
        used.update(group)
        pts = [(supports[j].x, supports[j].y) for j in group]
        if len(pts) == 1:
            shapes.append(Point(pts[0]).buffer(rad))
        else:
            pts_sorted = sorted(pts)
            shapes.append(
                LineString([pts_sorted[0], pts_sorted[-1]]).buffer(
                    rad, cap_style="round", join_style="round"
                )
            )
    merged = unary_union(shapes)
    if merged.geom_type == "Polygon":
        return [merged]  # type: ignore[list-item]
    return [g for g in getattr(merged, "geoms", []) if g.geom_type == "Polygon"]


def _draw_supports_and_loads(
    ax: Axes, problem: DesignProblem, load_case_index: int | None
) -> None:
    if problem.supports:
        min_dim = min(problem.width, problem.height)
        rad = max(max(float(s.tolerance) for s in problem.supports), 0.025 * min_dim)
        for poly in _support_bar_geometries(problem, rad):
            path = _polygon_to_compound_path(poly)
            if path is not None:
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor="#00f5ff",
                        edgecolor="#07080f",
                        linewidth=1.4,
                        alpha=1.0,
                        zorder=5,
                    )
                )

    cases = problem.effective_load_cases
    if load_case_index is not None:
        if load_case_index < 0 or load_case_index >= len(cases):
            raise ValueError(
                f"load_case_index {load_case_index} out of range for {len(cases)} load cases."
            )
        active_cases = [(load_case_index, cases[load_case_index])]
    else:
        active_cases = list(enumerate(cases))

    arrow_scale = 0.25 * min(problem.width, problem.height)
    for case_idx, loads in active_cases:
        color = _LOAD_CASE_COLORS[case_idx % len(_LOAD_CASE_COLORS)]
        for load in loads:
            mag = float(np.hypot(load.fx, load.fy))
            if mag <= 1e-9:
                continue
            dx = (load.fx / mag) * arrow_scale
            dy = (load.fy / mag) * arrow_scale
            ax.annotate(
                "",
                xy=(load.x + dx, load.y + dy),
                xytext=(load.x, load.y),
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.5,head_width=0.26",
                    "color": "#07080f",
                    "lw": 3.0,
                    "mutation_scale": 14,
                    "joinstyle": "miter",
                    "capstyle": "butt",
                    "alpha": 0.9,
                },
                zorder=6,
            )
            ax.annotate(
                "",
                xy=(load.x + dx, load.y + dy),
                xytext=(load.x, load.y),
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.5,head_width=0.26",
                    "color": color,
                    "lw": 1.7,
                    "mutation_scale": 14,
                    "joinstyle": "miter",
                    "capstyle": "butt",
                },
                zorder=7,
            )


def render_model_geometry_figure(
    problem: DesignProblem,
    cad_model: CADModel | None,
    density: np.ndarray,
    highlight_action: CutoutAction | None = None,
    load_case_index: int | None = None,
    title: str | None = None,
) -> Figure:
    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_dark_axis(ax)

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            problem.width,
            problem.height,
            facecolor=_BG_PLOT,
            edgecolor=_BORDER,
            linestyle="--",
            linewidth=0.9,
            zorder=0,
        )
    )

    if cad_model is not None:
        for poly in cad_model._extract_polygons(cad_model.current_geometry):
            path = _polygon_to_compound_path(poly)
            if path is not None:
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor=_MODEL_FILL,
                        edgecolor=_MODEL_EDGE,
                        linewidth=1.2,
                        alpha=0.82,
                        zorder=2,
                    )
                )
    else:
        ax.imshow(
            density,
            origin="lower",
            extent=(0.0, problem.width, 0.0, problem.height),
            cmap=_MODEL_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
            zorder=1,
        )

    _draw_keepouts(ax, problem)

    if highlight_action is not None and cad_model is not None:
        raw_cut = cad_model._action_to_geometry(highlight_action)
        clipped_cut = raw_cut.intersection(cad_model.stock_plate)
        if not cad_model.solid_keepouts.is_empty:
            clipped_cut = clipped_cut.difference(cad_model.solid_keepouts)
        for poly in cad_model._extract_polygons(clipped_cut):
            path = _polygon_to_compound_path(poly)
            if path is not None:
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor=(0.98, 0.44, 0.52, 0.25),
                        edgecolor="#fb7185",
                        linestyle="--",
                        linewidth=1.6,
                        zorder=5,
                    )
                )

    _draw_supports_and_loads(ax, problem, load_case_index)

    pad = 0.14 * max(problem.width, problem.height)
    ax.set_xlim(-pad * 0.5, problem.width + pad * 0.7)
    ax.set_ylim(-pad * 0.7, problem.height + pad * 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)

    vol = (
        cad_model.get_volume_fraction()
        if cad_model is not None
        else float(np.mean(density))
    )
    resolved_title = title or f"Resulting Model (Weight={vol:.3f})"
    ax.set_title(resolved_title, fontsize=9, fontweight="bold", color=_TEXT_PRIMARY)
    fig.tight_layout()
    return fig


def _field_range(
    raw_field: np.ndarray,
    density: np.ndarray,
    log_scale: bool,
) -> tuple[float, float]:
    void_mask = density <= 1e-3
    solid_values = raw_field[~void_mask]
    finite_values = solid_values[np.isfinite(solid_values)]
    if log_scale and finite_values.size > 0:
        positive_values = finite_values[finite_values > 0.0]
        if positive_values.size > 0:
            vmin = float(np.min(positive_values))
            vmax = float(np.max(positive_values))
            if vmax <= vmin * (1.0 + 1e-9):
                vmin = max(vmax * 0.1, 1e-30)
                vmax = max(vmax, vmin * 10.0)
            return vmin, vmax
        return 1e-6, 1.0
    if finite_values.size > 0:
        vmin = float(min(0.0, np.min(finite_values)))
        vmax = float(np.max(finite_values))
        if vmax <= vmin + 1e-12:
            vmax = vmin + 1.0
        return vmin, vmax
    return 0.0, 1.0


def compute_shared_field_ranges(
    items: Sequence[SolutionRecord | ReplayFrame],
    log_strain: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Computes shared (vmin, vmax) across solutions or replay frames for stress and strain energy."""
    if not items:
        return (0.0, 1.0), (1e-6 if log_strain else 0.0, 1.0)

    stress_ranges = [
        _field_range(item.fea_result.von_mises_stress, item.density, log_scale=False)
        for item in items
    ]
    strain_ranges = [
        _field_range(
            item.fea_result.element_strain_energies, item.density, log_scale=log_strain
        )
        for item in items
    ]
    stress_min = min(r[0] for r in stress_ranges)
    stress_max = max(r[1] for r in stress_ranges)
    if stress_max <= stress_min + 1e-12:
        stress_max = stress_min + 1.0

    strain_min = min(r[0] for r in strain_ranges)
    strain_max = max(r[1] for r in strain_ranges)
    if log_strain:
        strain_min = max(strain_min, 1e-30)
        if strain_max <= strain_min * (1.0 + 1e-9):
            strain_max = strain_min * 10.0
    elif strain_max <= strain_min + 1e-12:
        strain_max = strain_min + 1.0

    return (stress_min, stress_max), (strain_min, strain_max)


def compute_replay_field_ranges(
    frames: Sequence[ReplayFrame],
    log_strain: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Computes shared (vmin, vmax) across all replay frames for stress and strain energy."""
    return compute_shared_field_ranges(frames, log_strain=log_strain)


def render_fea_field_figure(
    problem: DesignProblem,
    density: np.ndarray,
    fea_result: FEAResult,
    field: FieldKind = "stress",
    cad_model: CADModel | None = None,
    log_scale: bool = False,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    if field == "stress":
        raw_field = fea_result.von_mises_stress
        cmap_name = "inferno"
        default_label = "Von Mises Stress"
    elif field == "strain_energy":
        raw_field = fea_result.element_strain_energies
        cmap_name = "plasma"
        default_label = "Element Strain Energy"
    else:
        raise ValueError(
            f"Unsupported field {field!r}; expected 'stress' or 'strain_energy'."
        )

    void_mask = density <= 1e-3
    masked_field = np.ma.masked_where(void_mask, raw_field)

    base_cmap = colormaps[cmap_name]
    cmap = base_cmap.with_extremes(bad=_BG_PLOT)

    solid_values = raw_field[~void_mask]
    finite_values = solid_values[np.isfinite(solid_values)]

    auto_vmin, auto_vmax = _field_range(raw_field, density, log_scale=log_scale)
    resolved_vmin = auto_vmin if vmin is None else float(vmin)
    resolved_vmax = auto_vmax if vmax is None else float(vmax)

    if log_scale:
        resolved_vmin = max(resolved_vmin, 1e-30)
        if resolved_vmax <= resolved_vmin * (1.0 + 1e-9):
            resolved_vmax = resolved_vmin * 10.0
        cmap = base_cmap.with_extremes(bad=_BG_PLOT, under=base_cmap(0.0))
        norm = LogNorm(vmin=resolved_vmin, vmax=resolved_vmax, clip=False)
    else:
        if resolved_vmax <= resolved_vmin + 1e-12:
            resolved_vmax = resolved_vmin + 1.0
        norm = Normalize(vmin=resolved_vmin, vmax=resolved_vmax)

    fig = _new_figure()
    ax = fig.add_subplot(111)
    _style_dark_axis(ax)

    im = ax.imshow(
        masked_field,
        origin="lower",
        extent=(0.0, problem.width, 0.0, problem.height),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    if cad_model is not None:
        for poly in cad_model._extract_polygons(cad_model.current_geometry):
            path = _polygon_to_compound_path(poly)
            if path is not None:
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor="none",
                        edgecolor="#f8fafc",
                        linewidth=0.75,
                        alpha=0.45,
                        zorder=3,
                    )
                )

    cbar_label = default_label
    if field == "stress" and problem.max_stress_limit is not None:
        cbar_label = f"{default_label} (limit <= {problem.max_stress_limit:.2f})"
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=7.5, color=_TEXT_MUTED)
    cbar.ax.tick_params(labelsize=7, colors="#ffffff")
    cbar.outline.set_edgecolor(_BORDER)  # type: ignore[attr-defined]

    peak_val = float(np.max(finite_values)) if finite_values.size > 0 else 0.0
    if title is None:
        if field == "stress":
            limit_note = (
                f" / {problem.max_stress_limit:.2f}"
                if problem.max_stress_limit is not None
                else ""
            )
            resolved_title = f"Von Mises Stress (Max={peak_val:.3f}{limit_note})"
        else:
            scale_tag = " [Log]" if log_scale else ""
            resolved_title = f"Strain Energy{scale_tag} (C={fea_result.compliance:.2f})"
    else:
        resolved_title = title

    ax.set_xlim(0.0, problem.width)
    ax.set_ylim(0.0, problem.height)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(resolved_title, fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def render_solution_views(
    solution_or_frame: SolutionRecord | ReplayFrame,
    problem: DesignProblem | None = None,
    load_case_index: int | None = None,
    log_strain: bool = False,
    highlight_cutout: bool = True,
    stress_range: tuple[float, float] | None = None,
    strain_range: tuple[float, float] | None = None,
) -> InspectionViews:
    if isinstance(solution_or_frame, SolutionRecord):
        target_problem = solution_or_frame.problem
        active_problem = problem or target_problem
        cad_model = solution_or_frame.cad_model
        density = solution_or_frame.density
        highlight = None
        label_prefix = solution_or_frame.label
    else:
        target_problem = solution_or_frame.cad_model.problem
        active_problem = problem or target_problem
        cad_model = solution_or_frame.cad_model
        density = solution_or_frame.density
        highlight = solution_or_frame.action if highlight_cutout else None
        label_prefix = f"Step {solution_or_frame.step}/{solution_or_frame.total_steps}"

    if (
        load_case_index is None
        and (problem is None or problem is target_problem)
        and len(solution_or_frame.fea_result.case_compliances)
        == len(target_problem.effective_load_cases)
    ):
        fea_res = solution_or_frame.fea_result
    else:
        fea_res = compute_load_case_fea(
            active_problem, density, load_case_index=load_case_index, penalty=1.0
        )

    case_tag = (
        f" [Case {load_case_index + 1}]"
        if load_case_index is not None
        else (" [All Cases]" if len(active_problem.effective_load_cases) > 1 else "")
    )

    model_fig = render_model_geometry_figure(
        problem=active_problem,
        cad_model=cad_model,
        density=density,
        highlight_action=highlight,
        load_case_index=load_case_index,
        title=f"{label_prefix} — Model{case_tag}",
    )
    stress_fig = render_fea_field_figure(
        problem=active_problem,
        density=density,
        fea_result=fea_res,
        field="stress",
        cad_model=cad_model,
        log_scale=False,
        title=f"Von Mises Stress{case_tag} (Max={float(np.max(fea_res.von_mises_stress)):.3f})",
        vmin=stress_range[0] if stress_range is not None else None,
        vmax=stress_range[1] if stress_range is not None else None,
    )
    strain_fig = render_fea_field_figure(
        problem=active_problem,
        density=density,
        fea_result=fea_res,
        field="strain_energy",
        cad_model=cad_model,
        log_scale=log_strain,
        title=f"Strain Energy{case_tag} (C={fea_res.compliance:.2f})",
        vmin=strain_range[0] if strain_range is not None else None,
        vmax=strain_range[1] if strain_range is not None else None,
    )

    return InspectionViews(
        model_figure=model_fig,
        stress_figure=stress_fig,
        strain_figure=strain_fig,
        fea_result=fea_res,
        density=density,
        load_case_index=load_case_index,
    )


def render_replay_trajectory_figure(
    frames: Sequence[ReplayFrame],
    problem: DesignProblem,
    active_step: int | None = None,
) -> Figure:
    fig = _new_figure(figsize=(8.0, 2.5), dpi=110)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax_vol, ax_comp, ax_stress = fig.subplots(1, 3)

    steps = [f.step for f in frames]
    vols = [f.volume_fraction for f in frames]
    comps = [f.compliance for f in frames]
    stresses = [f.max_von_mises_stress for f in frames]

    text_white = "#ffffff"
    spine_color = (1.0, 1.0, 1.0, 0.45)
    for axis in (ax_vol, ax_comp, ax_stress):
        axis.set_facecolor("none")
        axis.patch.set_alpha(0.0)
        axis.tick_params(colors=text_white, labelsize=7)
        axis.xaxis.label.set_color(text_white)
        axis.yaxis.label.set_color(text_white)
        axis.title.set_color(text_white)
        for spine in axis.spines.values():
            spine.set_color(spine_color)

    ax_vol.plot(steps, vols, marker="o", color="#22d3ee", linewidth=1.8, markersize=4)
    ax_vol.axhline(
        problem.target_volume_fraction,
        color="#facc15",
        linestyle="--",
        linewidth=1.1,
        label=f"Target ({problem.target_volume_fraction:.2f})",
    )
    ax_vol.set_title("Weight", fontsize=8.5, fontweight="bold")
    ax_vol.set_xlabel("Step", fontsize=7.5)
    ax_vol.legend(
        fontsize=7,
        loc="best",
        facecolor="none",
        edgecolor=spine_color,
        labelcolor=text_white,
    )

    ax_comp.plot(steps, comps, marker="o", color="#e879f9", linewidth=1.8, markersize=4)
    ax_comp.set_title("Compliance", fontsize=8.5, fontweight="bold")
    ax_comp.set_xlabel("Step", fontsize=7.5)

    ax_stress.plot(
        steps, stresses, marker="o", color="#fb923c", linewidth=1.8, markersize=4
    )
    if problem.max_stress_limit is not None:
        ax_stress.axhline(
            problem.max_stress_limit,
            color="#facc15",
            linestyle="--",
            linewidth=1.1,
            label=f"Limit ({problem.max_stress_limit:.2f})",
        )
        ax_stress.legend(
            fontsize=7,
            loc="best",
            facecolor="none",
            edgecolor=spine_color,
            labelcolor=text_white,
        )
    ax_stress.set_title("Max Von Mises Stress", fontsize=8.5, fontweight="bold")
    ax_stress.set_xlabel("Step", fontsize=7.5)

    if active_step is not None:
        for axis in (ax_vol, ax_comp, ax_stress):
            axis.axvline(active_step, color=text_white, linestyle=":", linewidth=1.1)

    fig.tight_layout()
    return fig
