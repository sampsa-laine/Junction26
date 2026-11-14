import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from .csg import CADModel
from .problem import HARBOR_TASK_BUILDERS, DesignProblem


AGENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENTS_DIR.parent.parent
TASKS_DIR = REPO_ROOT / "tasks"

SANDBOX_MODULES = ("problem.py", "fea.py", "csg.py", "export.py", "cad_cli.py")
VERIFIER_MODULES = ("problem.py", "fea.py", "csg.py", "simp.py", "verifier.py")
ORACLE_MODULES = ("problem.py", "fea.py", "csg.py", "simp.py", "oracle_solver.py")


TASK_METADATA = {
    "cantilever-lightweight": {
        "difficulty": "easy",
        "expert_min": 15.0,
        "junior_min": 45.0,
        "challenge_note": (
            "Primary bending tension/compression chords run along the top and bottom "
            "edges from the fixed left wall (x=0) down to the tip load boss at (57, 4), "
            "while the upper-right exterior corner and central web shear pockets carry "
            "minimal strain energy."
        ),
    },
    "l-bracket-stress": {
        "difficulty": "medium",
        "expert_min": 25.0,
        "junior_min": 60.0,
        "challenge_note": (
            "The void keep-out in [20, 50] x [20, 50] creates a sharp 90-degree "
            "re-entrant inner corner at (20, 20). Placing sharp cutouts or thinning the "
            "material near (20, 20) causes severe von Mises stress spikes above the "
            "allowable stress limit. Use smooth/filleted pockets away from high-stress load paths."
        ),
    },
    "suspension-rocker-multiload": {
        "difficulty": "hard",
        "expert_min": 35.0,
        "junior_min": 90.0,
        "challenge_note": (
            "This rocker arm connects 3 solid mounting bosses across TWO independent "
            "load cases (vertical bump compression + lateral braking shear). Optimizing "
            "for only one load case leaves struts that bend severely under the second "
            "load case. Preserve triangulated load paths serving both load vectors."
        ),
    },
}


def _render_task_toml(task_id: str, problem: DesignProblem) -> str:
    meta = TASK_METADATA[task_id]
    desc = problem.description.replace('"', '\\"')
    return f'''schema_version = "1.4"
artifacts = ["/app/design.json"]

[task]
name = "junction26/{task_id}"
version = "1.0.0"
description = "{desc}"
authors = [{{ name = "Junction26 Team", email = "team@junction26.local" }}]
keywords = ["cad", "fea", "topology-optimization", "mechanical-engineering", "simulation"]

[metadata]
author_name = "Junction26 Team"
author_email = "team@junction26.local"
difficulty = "{meta["difficulty"]}"
category = "scientific_computing"
tags = ["cad", "fea", "structural-optimization", "csg"]
expert_time_estimate_min = {meta["expert_min"]}
junior_time_estimate_min = {meta["junior_min"]}

[verifier]
timeout_sec = 180.0
environment_mode = "separate"
collect = []

[verifier.env]

[agent]
timeout_sec = 600.0

[environment]
network_mode = "public"
build_timeout_sec = 900.0
os = "linux"
mcp_servers = []

[environment.env]
PYTHONPATH = "/opt"
CAD_PROBLEM_PATH = "/app/problem.json"
CAD_DESIGN_PATH = "/app/design.json"

[solution.env]
PYTHONPATH = "/solution:/opt"
'''


def _render_instruction_md(task_id: str, problem: DesignProblem) -> str:
    meta = TASK_METADATA[task_id]
    cases_md = []
    for idx, case in enumerate(problem.effective_load_cases, start=1):
        vec_str = ", ".join(
            f"`(x={l.x:.1f}, y={l.y:.1f}, fx={l.fx:+.2f}, fy={l.fy:+.2f})`"
            for l in case
        )
        cases_md.append(f"  - **Load Case {idx}**: {vec_str}")
    cases_block = "\n".join(cases_md)

    keepouts_md = []
    for ko in problem.keepouts:
        keepouts_md.append(
            f"  - `{ko.keepout_type.value.upper()}` {ko.shape}: params `{tuple(round(v, 2) for v in ko.params)}`"
        )
    keepout_block = "\n".join(keepouts_md) if keepouts_md else "  - None"

    stress_str = (
        f"`<= {problem.max_stress_limit:.3f}`"
        if problem.max_stress_limit is not None
        else "Unconstrained"
    )

    return f"""# 2D Structural CAD Design Task: {problem.name}

## Objective
Engineer a lightweight, manufacturable 2D parametric CAD component saved at `/app/design.json`.
Starting from the stock plate in `/app/problem.json`, apply Constructive Solid Geometry (CSG) cutouts so the component meets the target mass budget while **minimizing structural compliance** (maximizing stiffness) across all load cases.

## Problem Specification & Engineering Constraints
- **Domain Bounding Box**: `{problem.width:.1f} x {problem.height:.1f}` (`{problem.nx} x {problem.ny}` Q4 finite element mesh)
- **Target Volume Fraction**: `<= {problem.target_volume_fraction:.3f}` (measured relative to `{problem.width:.0f} x {problem.height:.0f}` bounding box)
- **Maximum von Mises Stress Limit**: {stress_str}
- **Minimum Cutout Feature Radius / Half-Width**: `>= {problem.min_feature_radius:.2f}`
- **Minimum Solid Wall Thickness**: `>= {problem.min_wall_thickness:.2f}` (avoid paper-thin slivers or fragile bridges)
- **Maximum Parametric Cutouts**: `<= {problem.max_cutouts}` actions
- **External Load Cases**:
{cases_block}
- **Keep-Out Regions** (`SOLID` bosses cannot be cut; `VOID` zones are pre-cut):
{keepout_block}

### Engineering Note
{meta["challenge_note"]}

---

## Interactive Workbench Tools

### 1. Command-Line Tool (`cad-sim`)
Run `cad-sim` from anywhere to inspect the problem, apply CSG cutouts to `/app/design.json`, and run `<15ms` Q4 Finite Element Analysis (FEA):

```bash
# Inspect full specification & boundary conditions
cad-sim info

# Evaluate FEA compliance, stress hotspots, low-strain zones, and ASCII stress heatmap
cad-sim status --ascii

# Apply parametric CSG cutouts (automatically updates /app/design.json and runs FEA)
cad-sim cut-circle --x 30.0 --y 15.0 --r 4.5
cad-sim cut-slot --x1 18.0 --y1 12.0 --x2 38.0 --y2 12.0 --r 3.0
cad-sim cut-rect --x 20.0 --y 8.0 --w 12.0 --h 6.0 --fillet 2.0
cad-sim cut-polygon --points "25,28 55,28 55,14" --fillet 1.8

# Undo last cut or reset to uncut stock plate
cad-sim undo
cad-sim reset

# Optional: export CAD vector geometry
cad-sim export --svg /app/design.svg --dxf /app/design.dxf
```

### 2. Python Simulation API (`cad_env`)
You may also write and execute Python optimization/search scripts directly against the pre-installed `cad_env` package (`/opt/cad_env`):

```python
from cad_env import CADModel, DesignProblem, FEASolver, inspect_design

problem = DesignProblem.load_json("/app/problem.json")
solver = FEASolver(problem)
model = CADModel(problem)

model.cut_polygon([(25.0, 28.0), (56.0, 28.0), (56.0, 14.0)], fillet_radius=1.8)
model.cut_circle(22.0, 14.0, radius=4.0)

report = inspect_design(problem, model)
print(
    report["volume_fraction"],
    report["total_compliance"],
    report["max_von_mises_stress"],
)

# Always save your final CAD model to /app/design.json
model.save_json("/app/design.json")
```

---

## Scoring & Verification (`/logs/verifier/reward.txt`)
Your final `/app/design.json` is graded on a **continuous scale from `0.0` to `1.0`**:
- **Hard Validity Checks (`0.0` if violated)**:
  1. `/app/design.json` must exist, parse cleanly, remove at least `2%` area, use `<= {problem.max_cutouts}` cutouts, and obey `min_feature_radius >= {problem.min_feature_radius:.2f}`.
  2. A single connected solid structure must join the supports to **all** load points without floating solid islands or severed load paths.
- **Continuous Partial Credit (`0.0` to `1.0`)**:
  - Combines **stiffness efficiency** relative to the theoretical topological optimum ($C^*$), **volume budget satisfaction** (`1.0` when `volume_fraction <= {problem.target_volume_fraction:.3f} + 0.005`, smoothly penalized if over budget), **von Mises stress compliance** ({stress_str}), and **minimum wall thickness** (`>= {problem.min_wall_thickness:.2f}`).
"""


DOCKERFILE_CONTENT = """FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash \\
    ca-certificates \\
    curl \\
    git \\
    nodejs \\
    npm \\
    procps \\
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

RUN uv pip install --system --no-cache \\
    "numpy==2.5.4" \\
    "scipy==1.18.1" \\
    "shapely==2.2.0"

# Pre-install nvm + Node 22 + opencode-ai so Harbor agent setup is near-instant
ENV NVM_DIR=/root/.nvm
RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \\
    && . "$NVM_DIR/nvm.sh" \\
    && nvm install 22 \\
    && nvm alias default 22 \\
    && npm i -g opencode-ai@latest
ENV NPM_CONFIG_PREFER_OFFLINE=true

COPY cad_env /opt/cad_env
COPY cad-sim /usr/local/bin/cad-sim
RUN chmod +x /usr/local/bin/cad-sim

WORKDIR /app
COPY problem.json /app/problem.json
COPY design.json /app/design.json

ENV PYTHONPATH=/opt
ENV CAD_PROBLEM_PATH=/app/problem.json
ENV CAD_DESIGN_PATH=/app/design.json
"""

CAD_ENV_INIT = """from .cad_cli import inspect_design, load_workspace, render_ascii_map
from .csg import CADModel, CutoutAction
from .export import export_cad_to_dxf, export_cad_to_svg
from .fea import FEAResult, FEASolver
from .problem import DesignProblem, KeepOut, KeepOutType, Load, Support

__all__ = [
    "CADModel",
    "CutoutAction",
    "DesignProblem",
    "FEAResult",
    "FEASolver",
    "KeepOut",
    "KeepOutType",
    "Load",
    "Support",
    "export_cad_to_dxf",
    "export_cad_to_svg",
    "inspect_design",
    "load_workspace",
    "render_ascii_map",
]
"""

CAD_SIM_BIN = """#!/usr/bin/env python3
import sys
from cad_env.cad_cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""

TEST_SH_CONTENT = """#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
PYTHONPATH=/tests python3 -m verifier_pkg.verifier \\
    --problem /app/problem.json \\
    --design /app/design.json \\
    --reward-out /logs/verifier/reward.txt \\
    --metrics-out /logs/verifier/metrics.json
"""

SOLVE_SH_CONTENT = """#!/bin/bash
set -euo pipefail

PYTHONPATH=/solution:/opt python3 -m oracle_pkg.oracle_solver \\
    --problem /app/problem.json \\
    --design /app/design.json

cad-sim status --ascii
"""


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sync_harbor_tasks(output_root: Path = TASKS_DIR) -> list[Path]:
    """Generates standalone Harbor v1.4 task directories synced from src/agents/."""
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for task_id, builder in HARBOR_TASK_BUILDERS.items():
        problem = builder()
        task_dir = output_root / task_id
        env_dir = task_dir / "environment"
        cad_pkg_dir = env_dir / "cad_env"
        tests_dir = task_dir / "tests"
        ver_pkg_dir = tests_dir / "verifier_pkg"
        sol_dir = task_dir / "solution"
        ora_pkg_dir = sol_dir / "oracle_pkg"

        for d in (cad_pkg_dir, ver_pkg_dir, ora_pkg_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # 1. Task config & agent prompt
        (task_dir / "task.toml").write_text(_render_task_toml(task_id, problem))
        (task_dir / "instruction.md").write_text(
            _render_instruction_md(task_id, problem)
        )

        # 2. Sandbox environment (excludes simp.py, verifier.py, oracle_solver.py)
        (env_dir / "Dockerfile").write_text(DOCKERFILE_CONTENT)
        problem.save_json(env_dir / "problem.json")
        CADModel(problem).save_json(env_dir / "design.json")

        (cad_pkg_dir / "__init__.py").write_text(CAD_ENV_INIT)
        for mod in SANDBOX_MODULES:
            shutil.copy2(AGENTS_DIR / mod, cad_pkg_dir / mod)

        cad_sim_script = env_dir / "cad-sim"
        cad_sim_script.write_text(CAD_SIM_BIN)
        _make_executable(cad_sim_script)

        # 3. Verifier suite (uploaded to /tests after agent finishes)
        (ver_pkg_dir / "__init__.py").write_text("")
        for mod in VERIFIER_MODULES:
            shutil.copy2(AGENTS_DIR / mod, ver_pkg_dir / mod)

        test_sh = tests_dir / "test.sh"
        test_sh.write_text(TEST_SH_CONTENT)
        _make_executable(test_sh)

        # 4. Oracle reference solution
        (ora_pkg_dir / "__init__.py").write_text("")
        for mod in ORACLE_MODULES:
            shutil.copy2(AGENTS_DIR / mod, ora_pkg_dir / mod)

        solve_sh = sol_dir / "solve.sh"
        solve_sh.write_text(SOLVE_SH_CONTENT)
        _make_executable(solve_sh)

        generated.append(task_dir)

    return generated


def _load_dotenv_into_environ(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync and run Junction26 Harbor CAD tasks."
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser(
        "sync", help="Sync src/agents modules into tasks/* Harbor directories"
    )

    p_run = sub.add_parser(
        "run", help="Run a Harbor evaluation job (oracle, nop, or opencode)"
    )
    p_run.add_argument(
        "--task",
        default="cantilever-lightweight",
        choices=[*HARBOR_TASK_BUILDERS.keys(), "all"],
    )
    p_run.add_argument(
        "--agent", default="oracle", help="Agent name (oracle, nop, opencode)"
    )
    p_run.add_argument(
        "--model",
        default=None,
        help="Model slug for opencode, e.g. gemini/gemini-2.5-pro",
    )

    args = parser.parse_args(argv)
    cmd = args.cmd or "sync"

    generated = sync_harbor_tasks()
    if cmd == "sync":
        print(f"Synced {len(generated)} Harbor tasks from src/agents/:")
        for t_dir in generated:
            print(f"  - {t_dir.relative_to(REPO_ROOT)}")
        return 0

    _load_dotenv_into_environ(REPO_ROOT / ".env")
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get(
        "GOOGLE_GENERATIVE_AI_API_KEY"
    ):
        os.environ["GOOGLE_GENERATIVE_AI_API_KEY"] = os.environ["GEMINI_API_KEY"]
    elif os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") and not os.environ.get(
        "GEMINI_API_KEY"
    ):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]

    if args.agent == "opencode" and not args.model:
        args.model = os.environ.get("HARBOR_MODEL", "gemini/gemini-2.5-pro")

    if args.agent == "opencode" and not (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        print(
            "Error: Set GEMINI_API_KEY (or OPENAI_API_KEY) in .env before running --agent opencode.",
            file=sys.stderr,
        )
        return 1

    agent_env_flags: list[str] = []
    display_env_flags: list[str] = []
    if args.agent not in ("oracle", "nop"):
        agent_env_flags.extend(["--agent-setup-timeout-multiplier", "3.0"])
        display_env_flags.extend(["--agent-setup-timeout-multiplier", "3.0"])
        google_key = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if google_key:
            agent_env_flags.extend(
                [
                    "--ae",
                    f"GOOGLE_GENERATIVE_AI_API_KEY={google_key}",
                    "--ae",
                    f"GEMINI_API_KEY={google_key}",
                ]
            )
            display_env_flags.extend(
                [
                    "--ae",
                    "GOOGLE_GENERATIVE_AI_API_KEY=***",
                    "--ae",
                    "GEMINI_API_KEY=***",
                ]
            )
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            agent_env_flags.extend(["--ae", f"OPENAI_API_KEY={openai_key}"])
            display_env_flags.extend(["--ae", "OPENAI_API_KEY=***"])

    target_ids = (
        list(HARBOR_TASK_BUILDERS.keys()) if args.task == "all" else [args.task]
    )
    for tid in target_ids:
        base_cmd = [
            "uvx",
            "harbor",
            "run",
            "-p",
            str(TASKS_DIR / tid),
            "-a",
            args.agent,
        ]
        if args.model:
            base_cmd.extend(["-m", args.model])
        harbor_cmd = [*base_cmd, *agent_env_flags]
        print("Running:", " ".join([*base_cmd, *display_env_flags]))
        res = subprocess.run(harbor_cmd, cwd=REPO_ROOT)
        if res.returncode != 0:
            return res.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
