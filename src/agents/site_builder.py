from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from .csg import CADModel  # noqa: E402
from .problem import (  # noqa: E402
    DesignProblem,
    make_harbor_cantilever,
    make_harbor_l_bracket,
    make_harbor_suspension_rocker,
)
from .simp import SIMPResult  # noqa: E402
from .solutions import (  # noqa: E402
    ReplayFrame,
    SolutionRecord,
    build_human_solution,
    discover_harbor_runs,
    get_cached_reference_simp,
    load_harbor_trial,
    replay_cutout_history,
    run_in_repo_solver,
)
from .visualizer import render_solution_views  # noqa: E402

TASKS: list[tuple[str, str, DesignProblem]] = [
    (
        "cantilever-lightweight",
        "Cantilever Lightweight (60x30)",
        make_harbor_cantilever(),
    ),
    ("l-bracket-stress", "L-Bracket Stress Cap (50x50)", make_harbor_l_bracket()),
    (
        "suspension-rocker-multiload",
        "Suspension Rocker Multiload (80x40)",
        make_harbor_suspension_rocker(),
    ),
]


def _render_math_svg(expr: str, assets_dir: Path | None) -> str:
    """Render a LaTeX math expression to an SVG file via matplotlib.mathtext."""
    if assets_dir is None:
        return f"<code>{html.escape(expr)}</code>"

    assets_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(expr.encode("utf-8")).hexdigest()[:12]
    filename = f"math_{digest}.svg"
    out_path = assets_dir / filename

    if not out_path.exists():
        fig = plt.figure(figsize=(0.1, 0.1), dpi=150)
        fig.patch.set_alpha(0.0)
        fig.text(
            0.0,
            0.0,
            f"${expr}$",
            fontsize=13,
            color="#e2e8f0",
            usetex=False,
        )
        fig.savefig(
            out_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=True,
        )
        plt.close(fig)

    safe_alt = html.escape(expr, quote=True)
    return f'<img class="math-inline" src="assets/{filename}" alt="{safe_alt}" />'


def _inline_md(text: str, assets_dir: Path | None = None) -> str:
    """Convert inline Markdown (math, code, bold, italic, links) to HTML."""
    placeholders: dict[str, str] = {}

    def _stash_math(match: re.Match[str]) -> str:
        token = f"@@MATH{len(placeholders)}@@"
        placeholders[token] = _render_math_svg(match.group(1), assets_dir)
        return token

    with_tokens = re.sub(r"\$([^$\n]+)\$", _stash_math, text)
    escaped = html.escape(with_tokens)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener">\1</a>',
        escaped,
    )
    for token, replacement in placeholders.items():
        escaped = escaped.replace(token, replacement)
    return escaped


def markdown_to_html(md_text: str, assets_dir: Path | None = None) -> str:
    """Lightweight Markdown-to-HTML converter using only the Python stdlib."""
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(html.escape(lines[i]))
                i += 1
            i += 1
            cls = f' class="language-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>" + "\n".join(code_lines) + "</code></pre>")
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            out.append(
                f"<h{level}>{_inline_md(heading_match.group(2), assets_dir)}</h{level}>"
            )
            i += 1
            continue

        if stripped.startswith("> "):
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                quote_lines.append(_inline_md(lines[i].strip()[2:], assets_dir))
                i += 1
            out.append("<blockquote><p>" + " ".join(quote_lines) + "</p></blockquote>")
            continue

        if (
            "|" in stripped
            and i + 1 < len(lines)
            and re.match(r"^\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1].strip())
        ):
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            table_html = ['<div class="table-wrap"><table><thead><tr>']
            table_html.extend(f"<th>{_inline_md(h, assets_dir)}</th>" for h in headers)
            table_html.append("</tr></thead><tbody>")
            for row in rows:
                table_html.append("<tr>")
                table_html.extend(
                    f"<td>{_inline_md(cell, assets_dir)}</td>" for cell in row
                )
                table_html.append("</tr>")
            table_html.append("</tbody></table></div>")
            out.append("".join(table_html))
            continue

        if re.match(r"^[-*]\s+", stripped):
            out.append("<ul>")
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                out.append(f"<li>{_inline_md(item, assets_dir)}</li>")
                i += 1
            out.append("</ul>")
            continue

        para_lines: list[str] = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "```", "> ", "- ", "* "))
        ):
            para_lines.append(_inline_md(lines[i].strip(), assets_dir))
            i += 1
        out.append("<p>" + " ".join(para_lines) + "</p>")

    return "\n".join(out)


def _save_views(
    solution_or_frame: SolutionRecord | ReplayFrame,
    assets_dir: Path,
    prefix: str,
    problem: DesignProblem,
    skip_renders: bool = False,
) -> dict[str, str]:
    """Render geometry, stress, and strain views to PNG files (or reuse existing)."""
    if skip_renders:
        paths: dict[str, str] = {}
        for key in ("geometry", "stress", "strain"):
            filename = f"{prefix}_{key}.png"
            if (assets_dir / filename).exists():
                paths[key] = f"assets/{filename}"
        return paths

    views = render_solution_views(solution_or_frame, problem=problem)
    paths = {}
    for key, fig in [
        ("geometry", views.model_figure),
        ("stress", views.stress_figure),
        ("strain", views.strain_figure),
    ]:
        filename = f"{prefix}_{key}.png"
        out_path = assets_dir / filename
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[key] = f"assets/{filename}"
    return paths


def _read_trial_metadata(trial_dir: Path) -> tuple[str, str | None, float | None]:
    """Extract (agent_label, task_id, reward) directly from Harbor trial files."""
    cfg_path = trial_dir / "config.json"
    agent_label = trial_dir.name
    task_id: str | None = None

    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            agent_cfg = cfg.get("agent", {})
            name = str(agent_cfg.get("name") or "agent")
            model = str(agent_cfg.get("model_name") or "")
            agent_label = f"{name} ({model})" if model else name
            task_path = str((cfg.get("task") or {}).get("path") or "")
            if task_path:
                task_id = Path(task_path).name
        except Exception:
            pass

    if task_id is None and "__" in trial_dir.name:
        task_id = trial_dir.name.split("__", 1)[0]

    reward: float | None = None
    rew_path = trial_dir / "verifier" / "reward.txt"
    if rew_path.exists():
        try:
            reward = float(rew_path.read_text().strip())
        except ValueError:
            reward = None

    return agent_label, task_id, reward


def _serialize_solution(
    sol: SolutionRecord,
    images: dict[str, str] | None = None,
) -> dict[str, Any]:
    ver = sol.verification
    sb = sol.scoreboard
    return {
        "id": sol.solution_id,
        "label": sol.label,
        "source_kind": sol.source_kind,
        "solver_name": sol.solver_name,
        "task_id": sol.task_id,
        "harbor_reward": sol.harbor_reward,
        "arena_score": round(float(sb.overall_score), 2),
        "volume_fraction": round(float(sb.volume_fraction), 4),
        "max_stress": round(float(sb.max_von_mises_stress), 4),
        "compliance": round(float(sb.compliance), 4),
        "simp_gap": round(float(sb.optimality_gap_percent), 2),
        "connected": bool(ver.structurally_connected) if ver else True,
        "passed": bool(ver.valid) if ver else False,
        "images": images or {},
    }


def collect_site_data(
    jobs_root: Path,
    tasks_root: Path,
    assets_dir: Path,
    skip_renders: bool = False,
) -> dict[str, Any]:
    """Collect benchmarks, Harbor trials, baselines, and replay frames."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    simp_cache: dict[str, SIMPResult] = {}
    tasks_payload: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []

    oracle_by_task: dict[str, SolutionRecord] = {}

    for task_id, title, problem in TASKS:
        ref_simp = get_cached_reference_simp(problem)
        simp_cache[task_id] = ref_simp

        simp_sol = run_in_repo_solver(problem, "simp", reference_simp=ref_simp)
        oracle_sol = run_in_repo_solver(
            problem, "oracle_solver", reference_simp=ref_simp
        )
        oracle_by_task[task_id] = oracle_sol

        # Do-nothing (uncut stock) baseline evaluated with the real verifier
        uncut_cad = CADModel(problem)
        do_nothing_sol = build_human_solution(
            problem,
            uncut_cad,
            label="Do-Nothing Baseline (Uncut)",
            reference_simp=ref_simp,
        )

        simp_imgs = _save_views(
            simp_sol, assets_dir, f"{task_id}_simp", problem, skip_renders=skip_renders
        )
        oracle_imgs = _save_views(
            oracle_sol,
            assets_dir,
            f"{task_id}_oracle",
            problem,
            skip_renders=skip_renders,
        )

        oracle_replay: list[dict[str, Any]] = []
        if oracle_sol.cad_model is not None:
            frames = replay_cutout_history(oracle_sol.cad_model)
            for frame in frames:
                frame_imgs = _save_views(
                    frame,
                    assets_dir,
                    f"{task_id}_oracle_step_{frame.step:02d}",
                    problem,
                    skip_renders=skip_renders,
                )
                oracle_replay.append(
                    {
                        "step": frame.step,
                        "total_steps": frame.total_steps,
                        "action": frame.action.to_dict() if frame.action else None,
                        "volume_fraction": round(frame.volume_fraction, 4),
                        "compliance": round(frame.compliance, 4),
                        "max_stress": round(frame.max_von_mises_stress, 4),
                        "connected": frame.structurally_connected,
                        "images": frame_imgs,
                    }
                )

        tasks_payload.append(
            {
                "task_id": task_id,
                "title": title,
                "grid": f"{problem.nx}x{problem.ny}",
                "target_volume": problem.target_volume_fraction,
                "max_stress": problem.max_stress_limit,
                "load_cases": len(problem.effective_load_cases),
                "simp": _serialize_solution(simp_sol, simp_imgs),
                "oracle": _serialize_solution(oracle_sol, oracle_imgs),
                "do_nothing": _serialize_solution(do_nothing_sol),
                "oracle_replay": oracle_replay,
            }
        )

    # Discover Harbor runs in jobs/
    problem_map = {tid: prob for tid, _, prob in TASKS}
    trial_dirs = discover_harbor_runs(jobs_root) if jobs_root.exists() else []
    for trial_dir in trial_dirs:
        agent_label, inferred_task, raw_reward = _read_trial_metadata(trial_dir)
        run_entry: dict[str, Any] = {
            "trial": f"{trial_dir.parent.name}/{trial_dir.name}",
            "agent": agent_label,
            "task_id": inferred_task,
            "harbor_reward": raw_reward,
            "status": "completed" if raw_reward is not None else "pending",
            "images": {},
            "replay": [],
        }

        try:
            prob = problem_map.get(inferred_task or "")
            ref = simp_cache.get(inferred_task or "")
            sol = load_harbor_trial(
                trial_dir, problem=prob, tasks_root=tasks_root, reference_simp=ref
            )
            tid = sol.task_id or inferred_task or "unknown"
            run_entry["task_id"] = tid
            if sol.harbor_reward is not None:
                run_entry["harbor_reward"] = round(float(sol.harbor_reward), 4)
                run_entry["status"] = "completed"
            run_entry["arena_score"] = round(float(sol.scoreboard.overall_score), 2)
            run_entry["volume_fraction"] = round(
                float(sol.scoreboard.volume_fraction), 4
            )
            run_entry["max_stress"] = round(
                float(sol.scoreboard.max_von_mises_stress), 4
            )

            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", trial_dir.name)
            run_entry["images"] = _save_views(
                sol,
                assets_dir,
                f"{tid}_{safe_name}",
                sol.problem,
                skip_renders=skip_renders,
            )
            if sol.cad_model is not None and sol.cad_model.history:
                frames = replay_cutout_history(sol.cad_model)
                for frame in frames:
                    f_imgs = _save_views(
                        frame,
                        assets_dir,
                        f"{tid}_{safe_name}_step_{frame.step:02d}",
                        sol.problem,
                        skip_renders=skip_renders,
                    )
                    run_entry["replay"].append(
                        {
                            "step": frame.step,
                            "total_steps": frame.total_steps,
                            "action": (
                                frame.action.to_dict() if frame.action else None
                            ),
                            "volume_fraction": round(frame.volume_fraction, 4),
                            "compliance": round(frame.compliance, 4),
                            "max_stress": round(frame.max_von_mises_stress, 4),
                            "connected": frame.structurally_connected,
                            "images": f_imgs,
                        }
                    )
        except Exception as exc:
            run_entry["note"] = str(exc)

        all_runs.append(run_entry)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": tasks_payload,
        "harbor_runs": all_runs,
    }


def parse_hero_markdown(
    md_text: str, assets_dir: Path | None = None
) -> tuple[str, str]:
    """Split hero Markdown into copy HTML and <figure> HTML for standalone image lines."""
    copy_lines: list[str] = []
    figures: list[str] = []
    img_re = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")

    for line in md_text.splitlines():
        match = img_re.match(line.strip())
        if match:
            caption = match.group(1)
            src = html.escape(match.group(2).strip(), quote=True)
            safe_alt = html.escape(caption, quote=True)
            figures.append(
                f'<figure class="hero-thumb">'
                f'<img src="{src}" alt="{safe_alt}" />'
                f"<figcaption>{_inline_md(caption, assets_dir)}</figcaption>"
                f"</figure>"
            )
        else:
            copy_lines.append(line)

    copy_html = markdown_to_html("\n".join(copy_lines), assets_dir=assets_dir)
    return copy_html, "\n".join(figures)


def render_index_html(
    sections_html: dict[str, str],
    hero_visuals_html: str = "",
    data_json_rel: str = "data/results.json",
    repo_url: str = "https://github.com/sampsa-laine/Junction26",
) -> str:
    """Assemble the static site index.html from rendered Markdown sections."""
    safe_repo_url = html.escape(repo_url, quote=True)
    git_svg = (
        '<svg class="git-icon" viewBox="0 0 24 24" width="16" height="16" '
        'fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="8" r="2.5"/>'
        '<circle cx="6" cy="18" r="2.5"/><path d="M6 8.5v7"/>'
        '<path d="M18 10.5c0 4-6 3-12 5"/></svg>'
    )
    cache_bust = int(time.time())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Break the Agent — 2D Structural CAD Lightweighting</title>
  <link rel="stylesheet" href="styles.css?v={cache_bust}" />
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <div class="brand">
        <span class="brand-title">2D CAD TopOpt</span>
        <span class="team-badge" tabindex="0">
          by Qvabros
          <span class="team-popover" role="tooltip">
            <span>Rasmus Aaen</span>
            <span>Joel Kronqvist</span>
            <span>Sampsa Laine</span>
            <span>Rasmus Pouta</span>
          </span>
        </span>
      </div>
      <nav class="nav-links">
        <a class="nav-github" href="{safe_repo_url}" target="_blank" rel="noopener">{git_svg} Repo</a>
      </nav>
    </div>
  </header>

  <main class="container">
    <section id="overview" class="section hero-section">
      <div class="hero-grid">
        <div class="hero-copy">
          {sections_html.get("hero", "")}
        </div>
        <div id="hero-visuals" class="hero-visuals">
          {hero_visuals_html}
        </div>
      </div>
      <div class="hero-actions">
        <a class="repo-cta" href="{safe_repo_url}" target="_blank" rel="noopener">{git_svg} View Code &amp; Reproduce &nearr;</a>
      </div>
    </section>

    <nav class="toc-bar" aria-label="Table of Contents">
      <span class="toc-label">Contents</span>
      <a href="#gap">1. The Gap</a>
      <a href="#explorer">2. Visual &amp; Replay Explorer</a>
      <a href="#difficulty">3. Meaningful Difficulty</a>
      <a href="#integrity">4. Reward Integrity</a>
    </nav>

    <section id="gap" class="section">
      {sections_html.get("gap", "")}
      <div id="gap-chart" class="gap-chart"></div>
      <div id="runs-table-wrap" class="table-wrap"></div>
    </section>

    <section id="explorer" class="section">
      <div class="explorer-header">
        <div>
          <h2>Interactive Visual &amp; Replay Explorer</h2>
          <p class="muted">Compare SIMP Theoretical Optimum, Parametric Oracle, and Harbor Agent runs across all 3 tasks.</p>
        </div>
        <div class="controls">
          <label>
            Task
            <select id="task-select"></select>
          </label>
          <label>
            View
            <select id="view-select">
              <option value="geometry">Resulting CAD Geometry</option>
              <option value="stress">Von Mises Stress</option>
              <option value="strain">Element Strain Energy</option>
            </select>
          </label>
        </div>
      </div>

      <div id="comparison-grid" class="comparison-grid"></div>

      <div class="replay-box">
        <div class="replay-header">
          <div>
            <h3>Cutout History Replay (0 &rarr; N)</h3>
          </div>
          <label>
            Trajectory
            <select id="replay-source-select"></select>
          </label>
        </div>
        <div class="replay-controls">
          <button id="replay-prev" class="replay-btn" type="button" title="Previous step" aria-label="Previous step">&#8634;</button>
          <button id="replay-play" class="replay-btn replay-btn-primary" type="button" title="Play / Pause" aria-label="Play / Pause">&#9654;</button>
          <button id="replay-next" class="replay-btn" type="button" title="Next step" aria-label="Next step">&#8635;</button>
          <input id="replay-slider" type="range" min="0" max="0" value="0" />
          <span id="replay-step-label" class="muted">Step 0 / 0</span>
        </div>
        <div id="replay-metrics" class="stats-bar"></div>
        <div class="replay-mobile-toggle" role="tablist" aria-label="Replay view">
          <button type="button" role="tab" class="replay-tab active" data-mobile-view="geometry">Model</button>
          <button type="button" role="tab" class="replay-tab" data-mobile-view="stress">Stress</button>
        </div>
        <div id="replay-images" class="replay-images"></div>
      </div>
    </section>

    <section id="difficulty" class="section">
      {sections_html.get("difficulty", "")}
    </section>

    <section id="integrity" class="section">
      {sections_html.get("integrity", "")}
    </section>
  </main>

  <footer class="footer container">
    <p>Built by Qvabros for Junction 2026 &mdash; AfterQuery <em>Break the Agent</em> Challenge &middot; <a href="{safe_repo_url}" target="_blank" rel="noopener">Git Repository</a></p>
  </footer>

  <script>window.SITE_DATA_URL = "{data_json_rel}";</script>
  <script src="app.js?v={cache_bust}"></script>
</body>
</html>
"""


def build_site(
    site_dir: Path = Path("site"),
    jobs_root: Path = Path("jobs"),
    tasks_root: Path = Path("tasks"),
    skip_renders: bool = False,
    repo_url: str = "https://github.com/sampsa-laine/Junction26",
) -> Path:
    """Build the complete static site in `site_dir`."""
    content_dir = site_dir / "content"
    assets_dir = site_dir / "assets"
    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sections_html: dict[str, str] = {}
    hero_visuals_html = ""
    for md_file in sorted(content_dir.glob("*.md")):
        raw_md = md_file.read_text()
        if md_file.stem == "hero":
            copy_html, hero_visuals_html = parse_hero_markdown(
                raw_md, assets_dir=assets_dir
            )
            sections_html["hero"] = copy_html
        else:
            sections_html[md_file.stem] = markdown_to_html(
                raw_md, assets_dir=assets_dir
            )

    data = collect_site_data(
        jobs_root=jobs_root,
        tasks_root=tasks_root,
        assets_dir=assets_dir,
        skip_renders=skip_renders,
    )
    (data_dir / "results.json").write_text(json.dumps(data, indent=2))

    index_path = site_dir / "index.html"
    index_path.write_text(
        render_index_html(
            sections_html, hero_visuals_html=hero_visuals_html, repo_url=repo_url
        )
    )
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the static Junction26 showcase site from Markdown and Harbor jobs."
    )
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/sampsa-laine/Junction26",
        help="GitHub repository URL linked in header and footer.",
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Skip Matplotlib PNG rendering for fast Markdown-only rebuilds.",
    )
    args = parser.parse_args()

    out = build_site(
        site_dir=args.site_dir,
        jobs_root=args.jobs_dir,
        tasks_root=args.tasks_dir,
        skip_renders=args.skip_renders,
        repo_url=args.repo_url,
    )
    print(f"Built static site: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
