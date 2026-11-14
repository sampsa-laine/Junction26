import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .csg import CADModel
from .export import export_cad_to_dxf, export_cad_to_svg
from .fea import FEASolver
from .problem import DesignProblem, KeepOutType


DEFAULT_PROBLEM_PATH = "/app/problem.json"
DEFAULT_DESIGN_PATH = "/app/design.json"


def load_workspace(
    problem_path: str | Path | None = None,
    design_path: str | Path | None = None,
) -> tuple[DesignProblem, CADModel, Path]:
    p_path = Path(
        problem_path or os.environ.get("CAD_PROBLEM_PATH", DEFAULT_PROBLEM_PATH)
    )
    d_path = Path(design_path or os.environ.get("CAD_DESIGN_PATH", DEFAULT_DESIGN_PATH))
    problem = DesignProblem.load_json(p_path)
    if d_path.exists():
        model = CADModel.load_json(d_path, problem)
    else:
        model = CADModel(problem)
    return problem, model, d_path


def render_ascii_map(
    problem: DesignProblem,
    density: np.ndarray,
    stress: np.ndarray,
    max_cols: int = 60,
    max_rows: int = 24,
) -> str:
    ny, nx = density.shape
    step_x = max(1, int(np.ceil(nx / max_cols)))
    step_y = max(1, int(np.ceil(ny / max_rows)))

    dx = problem.width / nx
    dy = problem.height / ny
    max_s = max(float(np.max(stress)), 1e-9)
    levels = "123456789#"

    lines = [
        f"  Y ^  Stress Map ('.'=void, '1'..'9','#'=low..peak stress, 'K'=solid keepout)"
    ]
    row_indices = list(range(0, ny, step_y))[::-1]
    col_indices = list(range(0, nx, step_x))

    for j in row_indices:
        y_val = (j + 0.5) * dy
        chars: list[str] = []
        for i in col_indices:
            x_val = (i + 0.5) * dx
            sub_d = float(
                np.mean(density[j : min(ny, j + step_y), i : min(nx, i + step_x)])
            )
            sub_s = float(
                np.max(stress[j : min(ny, j + step_y), i : min(nx, i + step_x)])
            )

            in_solid_ko = False
            for ko in problem.keepouts:
                if ko.keepout_type != KeepOutType.SOLID:
                    continue
                if ko.shape == "circle":
                    cx, cy, r = ko.params
                    if np.hypot(x_val - cx, y_val - cy) <= r:
                        in_solid_ko = True
                        break
                elif ko.shape == "rect":
                    rx, ry, rw, rh = ko.params
                    if rx <= x_val <= rx + rw and ry <= y_val <= ry + rh:
                        in_solid_ko = True
                        break

            if sub_d < 0.25:
                chars.append(".")
            elif in_solid_ko:
                chars.append("K")
            else:
                idx = int(
                    np.clip(np.floor((sub_s / max_s) * len(levels)), 0, len(levels) - 1)
                )
                chars.append(levels[idx])
        lines.append(f"{y_val:5.1f} | {''.join(chars)}")

    lines.append("      +" + "-" * len(col_indices))
    x_end = problem.width
    pad = max(0, len(col_indices) - 10)
    lines.append(f"        0.0{' ' * pad}{x_end:5.1f} -> X")
    return "\n".join(lines)


def inspect_design(problem: DesignProblem, model: CADModel) -> dict[str, Any]:
    solver = FEASolver(problem)
    density = model.to_density_grid()
    fea_res = solver.solve(density, penalty=1.0)

    vol = model.get_volume_fraction()
    stock_vol = model.get_stock_volume_fraction()
    connected = model.is_structurally_connected()
    wall_ok, sliver_ratio = model.check_wall_thickness()
    feature_violations = model.check_feature_constraints()
    max_vm = float(np.max(fea_res.von_mises_stress))

    dx = problem.width / problem.nx
    dy = problem.height / problem.ny
    xs = (np.arange(problem.nx) + 0.5) * dx
    ys = (np.arange(problem.ny) + 0.5) * dy

    solid_mask = density > 0.6
    stress = fea_res.von_mises_stress
    strain = fea_res.element_strain_energies

    hotspots: list[dict[str, float]] = []
    coldspots: list[dict[str, float]] = []

    if np.any(solid_mask):
        high_s = np.where(solid_mask, stress, -np.inf).flatten()
        for idx in np.argsort(high_s)[-5:][::-1]:
            if high_s[idx] > -np.inf:
                hotspots.append(
                    {
                        "x": round(float(xs[idx % problem.nx]), 2),
                        "y": round(float(ys[idx // problem.nx]), 2),
                        "stress": round(float(high_s[idx]), 4),
                    }
                )

        low_e = np.where(solid_mask, strain, np.inf).flatten()
        for idx in np.argsort(low_e)[:8]:
            if low_e[idx] < np.inf:
                coldspots.append(
                    {
                        "x": round(float(xs[idx % problem.nx]), 2),
                        "y": round(float(ys[idx // problem.nx]), 2),
                        "strain_energy": round(float(low_e[idx]), 6),
                        "stress": round(float(stress.flatten()[idx]), 4),
                    }
                )

    return {
        "problem_name": problem.name,
        "num_actions": len(model.history),
        "max_cutouts": problem.max_cutouts,
        "volume_fraction": round(vol, 4),
        "stock_volume_fraction": round(stock_vol, 4),
        "target_volume_fraction": round(problem.target_volume_fraction, 4),
        "volume_budget_met": vol <= problem.target_volume_fraction + 0.005,
        "total_compliance": round(fea_res.compliance, 4),
        "case_compliances": [round(c, 4) for c in fea_res.case_compliances],
        "max_von_mises_stress": round(max_vm, 4),
        "max_stress_limit": problem.max_stress_limit,
        "stress_limit_met": (
            True
            if problem.max_stress_limit is None
            else max_vm <= problem.max_stress_limit
        ),
        "structurally_connected": connected,
        "wall_thickness_ok": wall_ok,
        "thin_sliver_ratio": round(sliver_ratio, 4),
        "feature_violations": feature_violations,
        "hotspots": hotspots,
        "coldspots": coldspots,
        "_density": density,
        "_stress": stress,
    }


def _format_duration(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    mins, rem = divmod(secs, 60)
    return f"{mins}m {rem:02d}s"


def _get_time_budget_lines() -> list[str]:
    hard_limit = float(os.environ.get("CAD_TIME_LIMIT_SEC", "600"))
    target_finish = float(os.environ.get("CAD_TARGET_FINISH_SEC", "540"))

    env_start = os.environ.get("CAD_SESSION_START_TS")
    if env_start is not None:
        start_ts = float(env_start)
    else:
        marker = Path(tempfile.gettempdir()) / f".cad_sim_start_{os.getuid()}.ts"
        if marker.exists():
            try:
                start_ts = float(marker.read_text().strip())
            except ValueError:
                start_ts = time.time()
                marker.write_text(f"{start_ts:.3f}")
        else:
            start_ts = time.time()
            marker.write_text(f"{start_ts:.3f}")

    elapsed = max(0.0, time.time() - start_ts)
    rem_target = max(0.0, target_finish - elapsed)
    lines = [
        f"Time Budget     : {_format_duration(elapsed)} elapsed | "
        f"Finish target <= {_format_duration(target_finish)} ({_format_duration(rem_target)} left) | "
        f"Hard limit {_format_duration(hard_limit)}"
    ]
    if elapsed >= target_finish - 60.0:
        lines.append(
            "WARNING: Approaching/past 9-minute finish target — finalize /app/design.json and stop now!"
        )
    return lines


def _print_summary(
    problem: DesignProblem, report: dict[str, Any], show_ascii: bool
) -> None:
    print(
        f"=== {problem.name} | Actions: {report['num_actions']}/{report['max_cutouts']} ==="
    )
    for line in _get_time_budget_lines():
        print(line)
    print(
        f"Volume Fraction : {report['volume_fraction']:.4f} "
        f"(Target <= {report['target_volume_fraction']:.4f}, Stock = {report['stock_volume_fraction']:.4f}) "
        f"[{'OK' if report['volume_budget_met'] else 'OVER BUDGET'}]"
    )
    print(
        f"Compliance      : {report['total_compliance']:.4f} "
        f"(Per load case: {report['case_compliances']})"
    )
    stress_lim = (
        f"<= {report['max_stress_limit']:.3f}"
        if report["max_stress_limit"] is not None
        else "unconstrained"
    )
    print(
        f"Max von Mises   : {report['max_von_mises_stress']:.4f} ({stress_lim}) "
        f"[{'OK' if report['stress_limit_met'] else 'EXCEEDS LIMIT'}]"
    )
    print(
        f"Connected       : {'YES' if report['structurally_connected'] else 'NO (SEVERED LOAD PATH!)'} | "
        f"Wall Thickness (>= {problem.min_wall_thickness:.2f}): "
        f"{'OK' if report['wall_thickness_ok'] else f'THIN SLIVERS ({report["thin_sliver_ratio"]:.3f})'}"
    )
    if report["feature_violations"]:
        print(f"Feature Errors  : {report['feature_violations']}")
    if report["hotspots"]:
        hs_str = ", ".join(
            f"({h['x']}, {h['y']}): {h['stress']:.3f}" for h in report["hotspots"][:4]
        )
        print(f"Peak Stress At  : {hs_str}")
    if report["coldspots"]:
        cs_str = ", ".join(f"({c['x']}, {c['y']})" for c in report["coldspots"][:5])
        print(f"Low-Strain Zones: {cs_str}")
    if show_ascii:
        print()
        print(render_ascii_map(problem, report["_density"], report["_stress"]))


def _parse_polygon_points(raw: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for token in raw.replace(";", " ").split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid coordinate pair '{token}', expected 'x,y'")
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cad-sim",
        description="2D Parametric CAD & FEA Structural Workbench",
    )
    parser.add_argument("--problem", default=None, help="Path to problem.json")
    parser.add_argument("--design", default=None, help="Path to design.json")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Display problem specification and constraints")

    p_status = sub.add_parser("status", help="Run FEA and inspect current design")
    p_status.add_argument("--ascii", action="store_true", help="Print ASCII stress map")
    p_status.add_argument("--json", action="store_true", help="Output raw JSON metrics")

    p_circle = sub.add_parser("cut-circle", help="Cut a circular hole")
    p_circle.add_argument("--x", type=float, required=True)
    p_circle.add_argument("--y", type=float, required=True)
    p_circle.add_argument("--r", type=float, required=True)
    p_circle.add_argument("--mirror-x", action="store_true")
    p_circle.add_argument("--mirror-y", action="store_true")

    p_rect = sub.add_parser(
        "cut-rect", help="Cut a rectangle or rounded-rectangle pocket"
    )
    p_rect.add_argument("--x", type=float, required=True)
    p_rect.add_argument("--y", type=float, required=True)
    p_rect.add_argument("--w", type=float, required=True)
    p_rect.add_argument("--h", type=float, required=True)
    p_rect.add_argument("--fillet", type=float, default=0.0)
    p_rect.add_argument("--mirror-x", action="store_true")
    p_rect.add_argument("--mirror-y", action="store_true")

    p_slot = sub.add_parser("cut-slot", help="Cut a rounded linear slot")
    p_slot.add_argument("--x1", type=float, required=True)
    p_slot.add_argument("--y1", type=float, required=True)
    p_slot.add_argument("--x2", type=float, required=True)
    p_slot.add_argument("--y2", type=float, required=True)
    p_slot.add_argument("--r", type=float, required=True)
    p_slot.add_argument("--mirror-x", action="store_true")
    p_slot.add_argument("--mirror-y", action="store_true")

    p_poly = sub.add_parser("cut-polygon", help="Cut a polygonal truss pocket")
    p_poly.add_argument(
        "--points",
        required=True,
        help="Space-separated 'x,y' vertices, e.g. '10,5 30,5 20,20'",
    )
    p_poly.add_argument("--fillet", type=float, default=0.0)
    p_poly.add_argument("--mirror-x", action="store_true")
    p_poly.add_argument("--mirror-y", action="store_true")

    sub.add_parser("undo", help="Undo most recent cutout")
    sub.add_parser("reset", help="Reset design to uncut stock plate")

    p_export = sub.add_parser("export", help="Export CAD design to SVG and/or DXF")
    p_export.add_argument("--svg", default=None, help="Output SVG path")
    p_export.add_argument("--dxf", default=None, help="Output DXF path")

    args = parser.parse_args(argv)
    problem, model, d_path = load_workspace(args.problem, args.design)

    if args.cmd == "info":
        spec = {
            "name": problem.name,
            "description": problem.description,
            "domain_size": [problem.width, problem.height],
            "mesh_grid": [problem.nx, problem.ny],
            "target_volume_fraction": problem.target_volume_fraction,
            "max_stress_limit": problem.max_stress_limit,
            "min_feature_radius": problem.min_feature_radius,
            "min_wall_thickness": problem.min_wall_thickness,
            "max_cutouts": problem.max_cutouts,
            "num_supports": len(problem.supports),
            "support_bounds_x": [
                min(s.x for s in problem.supports),
                max(s.x for s in problem.supports),
            ],
            "support_bounds_y": [
                min(s.y for s in problem.supports),
                max(s.y for s in problem.supports),
            ],
            "load_cases": [
                [{"x": l.x, "y": l.y, "fx": l.fx, "fy": l.fy} for l in case]
                for case in problem.effective_load_cases
            ],
            "keepouts": [
                {
                    "type": k.keepout_type.value,
                    "shape": k.shape,
                    "params": list(k.params),
                }
                for k in problem.keepouts
            ],
            "design_file": str(d_path),
            "current_actions": len(model.history),
        }
        print(json.dumps(spec, indent=2))
        return 0

    if args.cmd == "status":
        report = inspect_design(problem, model)
        if args.json:
            clean = {k: v for k, v in report.items() if not k.startswith("_")}
            print(json.dumps(clean, indent=2))
        else:
            _print_summary(problem, report, show_ascii=args.ascii)
        return 0

    if args.cmd == "reset":
        model.reset()
        model.save_json(d_path)
        report = inspect_design(problem, model)
        _print_summary(problem, report, show_ascii=False)
        return 0

    if args.cmd == "undo":
        if not model.undo():
            print("Nothing to undo.", file=sys.stderr)
            return 1
        model.save_json(d_path)
        report = inspect_design(problem, model)
        _print_summary(problem, report, show_ascii=False)
        return 0

    if args.cmd == "export":
        if args.svg:
            Path(args.svg).write_text(export_cad_to_svg(model))
            print(f"Wrote SVG to {args.svg}")
        if args.dxf:
            Path(args.dxf).write_text(export_cad_to_dxf(model))
            print(f"Wrote DXF to {args.dxf}")
        return 0

    ok = False
    if args.cmd == "cut-circle":
        ok = model.cut_circle(args.x, args.y, args.r, args.mirror_x, args.mirror_y)
    elif args.cmd == "cut-rect":
        ok = model.cut_rect(
            args.x, args.y, args.w, args.h, args.fillet, args.mirror_x, args.mirror_y
        )
    elif args.cmd == "cut-slot":
        ok = model.cut_slot(
            args.x1, args.y1, args.x2, args.y2, args.r, args.mirror_x, args.mirror_y
        )
    elif args.cmd == "cut-polygon":
        pts = _parse_polygon_points(args.points)
        ok = model.cut_polygon(pts, args.fillet, args.mirror_x, args.mirror_y)

    if not ok:
        print(
            "Cut rejected (outside domain, empty, or blocked by solid keepout).",
            file=sys.stderr,
        )
        return 1

    model.save_json(d_path)
    report = inspect_design(problem, model)
    _print_summary(problem, report, show_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
