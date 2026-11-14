# 2D CAD Topology Optimization Arena & Inspector

Frontend and CLI environment for evaluating, comparing, and replaying 2D structural CAD designs across Harbor benchmark runs and in-repo solvers.

## How to Run

| Target | Command | Purpose |
| :--- | :--- | :--- |
| **Web Inspector & Arena** | `uv run arena` | Launches Streamlit multi-model evaluator, FEA stress/strain viewer, and cutout history replay (`src/agents/app.py`) |
| **CLI Benchmark** | `uv run junction26` | Runs terminal comparison benchmark |
| **Harbor Task Sync** | `uv run harbor-tasks sync` | Syncs sandbox, verifier, and oracle modules into `tasks/*` |

## Supported Solution Sources

| Source | Identifier / Input | Output Geometry | Cutout Replay |
| :--- | :--- | :--- | :--- |
| **Harbor Trials** | `jobs/<job>/<trial>/` or `design.json` | Parametric CSG (`CADModel`) | Steps `0..N` |
| **SIMP** | `simp` (`SIMPOptimizer`) | Continuous `(ny, nx)` density | N/A (continuous) |
| **Oracle Solver** | `oracle_solver` (`ParametricCADOptimizer`) | Parametric CSG (`CADModel`) | Steps `0..N` |
| **Heuristic Agent** | `heuristic_agent` (`HeuristicCADDesigner`) | Parametric CSG (`CADModel`) | Steps `0..N` |
| **LLM Designer** | `llm_designer` (`LLMCADDesigner`) | Parametric CSG (`CADModel`) + ReAct log | Steps `0..N` |
| **Human Canvas** | Interactive Streamlit tools | Parametric CSG (`CADModel`) | Steps `0..N` |

## Inspector Tabs & Views

| View | Features |
| :--- | :--- |
| **Solution Comparison & Inspection** | Live stock problem preview; unified Harbor reward (`[0, 1]`) & Arena score (`[0, 100]`) table filtered to the active problem by default; side-by-side **Resulting Model**, **Von Mises Stress**, and **Element Strain Energy** plots (`penalty=1.0`); multi-load envelope vs per-case selector |
| **Cutout History Replay (`0..N`)** | Automatically syncs to the active problem or latest solver/trial run; reconstructs saved `CADModel` history from uncut stock (`step 0`) to final design (`step N`), highlighting active cutouts (`circle`, `rect`, `slot`, `polygon`, fillets, mirrors) with step FEA fields and `0..N` metric trajectories |
| **Human Canvas & LLM Log** | Manual `circle` / `rect` / `slot` cutouts with live FEA plots and LLM ReAct step transcripts |
| **CAD Export** | Downloads any loaded or replayed `CADModel` as SVG, AutoCAD R12 DXF, or `design.json` |

## Module Layout (`src/agents/`)

- `problem.py`: Domains, supports, multi-load cases, keepouts, constraints, JSON I/O, Harbor & Arena presets.
- `fea.py`: Vectorized Q4 linear elastic FEA solver (`von_mises_stress`, `element_strain_energies`, `case_compliances`).
- `simp.py`: Multi-load SIMP topology optimizer ($C^*$).
- `csg.py`: Shapely CSG `CADModel` and `CutoutAction` serialization (`design.json`).
- `solutions.py`: Harbor trial discovery/loading, automatic `DesignProblem` resolution, in-repo solver execution with cached `SIMPResult`, comparison table builder, and `0..N` cutout history replay.
- `visualizer.py`: Matplotlib OO `Figure` renderers for resulting model geometry, void-masked von Mises stress, element strain energy (envelope & per-load-case), and replay trajectories.
- `app.py`: Streamlit evaluation, comparison, and cutout replay dashboard (`uv run arena`).
- `cad_cli.py`: Sandbox `cad-sim` CLI and `cad_env` SDK.
- `verifier.py`: Continuous `[0.0, 1.0]` Harbor reward scorer.
- `oracle_solver.py`: Reference SIMP-to-CAD parametric fitter.
- `heuristic_agent.py`: Strain-energy guided baseline designer.
- `llm_designer.py`: ReAct CAD agent loop.
- `metrics.py`: Arena composite scoreboard (`[0, 100]`, IoU, SIMP gap).
- `export.py`: DXF and SVG vector exporters.
- `harbor_sync.py`: Standalone Harbor v1.4 task generator and runner.
- `main.py`: CLI benchmark and Streamlit launcher.
