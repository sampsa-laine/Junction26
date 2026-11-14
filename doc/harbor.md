# Harbor Evaluation Suite

Three graded Harbor v1.4 tasks benchmark coding agents (e.g., OpenCode) on 2D structural CAD lightweighting with `<15ms` Q4 FEA feedback. All implementation modules live in `src/agents/` and sync into `tasks/` via `uv run harbor-tasks sync`.

## Task Suite (`tasks/`)

| Task Directory | Difficulty | Domain & Mesh | Target Vol | Key Challenge | Oracle Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cantilever-lightweight` | Easy | `60x30` (`60x30`) | `<= 0.45` | Truss triangulation between fixed wall (`x=0`) and tip boss `(57, 4)` | `~0.95` |
| `l-bracket-stress` | Medium | `50x50` (`50x50`) | `<= 0.36` | Sharp `(20, 20)` re-entrant corner with `von Mises <= 1.85` stress cap | `~0.97` |
| `suspension-rocker-multiload` | Hard | `80x40` (`64x32`) | `<= 0.40` | 3 solid mounting bosses across **2 independent load cases** (bump + braking) | `~0.94` |
| `aerospace-rib-obstacles` | Hard | `80x40` (`64x32`) | `<= 0.35` | 3 solid lugs + **2 void pass-through tunnels** across **3 independent load cases** | `~0.92+` |

## Architecture & Anti-Cheat Separation

- **Sandbox (`environment/`)**: Exposes `cad-sim` CLI and `cad_env` Python package (`problem.py`, `fea.py`, `csg.py`, `export.py`, `cad_cli.py`). Excludes `simp.py`, `verifier.py`, and `oracle_solver.py`.
- **Verifier (`tests/`, `environment_mode = "separate"`)**: Collects `/app/design.json` into a clean container, checks connectivity, feature limits, wall thickness, volume budget, and stress caps against `SIMPOptimizer` ($C^*$), and writes a continuous reward $\in [0.0, 1.0]$ to `/logs/verifier/reward.txt`.
- **Oracle (`solution/`)**: Fits filleted CSG polygons and holes to SIMP void cavities (`oracle_solver.py`).

## Commands

```shell
# Sync tasks/* from src/agents/ and lint
uv run harbor-tasks sync
uvx harbor lint tasks/*

# Run Oracle validation in Docker (requires running Docker daemon)
uv run harbor-tasks run --task all --agent oracle

# Run OpenCode with Gemini (reads GEMINI_API_KEY from .env)
uv run harbor-tasks run --task cantilever-lightweight --agent opencode --model gemini/gemini-2.5-pro
```
