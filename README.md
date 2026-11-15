# Junction26 — 2D Structural Design Benchmark

Reinforcement learning and Harbor v1.4 benchmark environment for 2D structural CAD lightweighting with `<15ms` Q4 finite element analysis (FEA) feedback, scored against continuous topological optimization ($C^*$).

## Quick start

```shell
# 1. Install dependencies (requires Python >= 3.12 and uv)
uv sync

# 2. Optional: configure API keys for OpenCode / Gemini agent runs
cp .env.example .env

# 3. Launch the interactive Arena & Inspector
uv run arena
```

## Commands available to CAD agent

| Command | Purpose |
| --- | --- |
| `cad-sim info` | Show problem specification and constraints as JSON |
| `cad-sim status [--ascii] [--json]` | Run FEA; show metrics, an optional stress map, or JSON |
| `cad-sim cut-circle --x X --y Y --r R` | Cut a circular hole |
| `cad-sim cut-rect --x X --y Y --w W --h H [--fillet R]` | Cut a rectangular or rounded pocket |
| `cad-sim cut-slot --x1 X1 --y1 Y1 --x2 X2 --y2 Y2 --r R` | Cut a rounded slot |
| `cad-sim cut-polygon --points "x,y x,y x,y …" [--fillet R]` | Cut a polygonal pocket |
| `cad-sim undo` / `cad-sim reset` | Undo the last cut / restore stock |
| `cad-sim export [--svg PATH] [--dxf PATH]` | Export vector geometry |

All cuts accept `--mirror-x` and `--mirror-y`, save automatically, and run FEA.
Feedback includes volume budget, total and per-load-case compliance, stress limits,
connectivity, wall thickness, feature violations, hotspots, and low-strain zones.

## Command cheatsheet

| Command | Purpose |
| --- | --- |
| `uv sync` | Install project and development dependencies |
| `uv run arena` | Launch interactive Streamlit Arena & Inspector |
| `uv run junction26` | Run terminal comparison benchmark |
| `uv run harbor-tasks sync` | Sync `tasks/*` from `src/agents/` |
| `uv run harbor-tasks run --task all --agent oracle` | Run reference implementation across all 4 tasks |
| `uv run harbor-tasks run --task l-bracket-stress --agent opencode` | Run OpenCode agent (reads `GEMINI_API_KEY` & `HARBOR_MODEL` from `.env`) |
| `uv run build-site` | Build static showcase site in `site/` |
| `uv run pytest` | Run `unittest` tests and doctests |

## Running Harbor evaluations

Harbor runs execute inside Docker by default (requires a running Docker daemon) or remotely via Modal (`--env modal`):

```shell
# Sync sandbox, verifier, and reference modules into tasks/* and lint
uv run harbor-tasks sync
uvx harbor lint tasks/*

# Reproduce reference implementation across all 4 tasks
uv run harbor-tasks run --task all --agent oracle

# Run OpenCode on a single task (or use --task all)
uv run harbor-tasks run --task suspension-rocker-multiload --agent opencode
```

Available tasks: `cantilever-lightweight`, `l-bracket-stress`, `suspension-rocker-multiload`, `aerospace-rib-obstacles`.

## Building the static site

```shell
# Full build (renders FEA/CAD plots from jobs/ + tasks/ and builds site/index.html)
uv run build-site

# Fast rebuild after editing site/content/*.md (reuses existing PNGs)
uv run build-site --skip-renders

# Preview locally
python3 -m http.server --directory site 8000
```

## Documentation & project layout

- [Table of Contents](doc/table-of-contents.md)
- [Arena & Inspector](doc/arena.md)
- [Harbor Evaluation Suite](doc/harbor.md)
- [Project Layout & Testing](doc/project-layout.md)

New CLI targets can be added under `[project.scripts]` in [pyproject.toml](pyproject.toml) (for example, `junction26 = "src.agents.main:main"`).

## Using Modal to offload tasks

Set up your Modal API key with `uv run modal setup`, then pass `--env modal` to `uv run harbor-tasks run ...` or `uv run harbor run ...`.
