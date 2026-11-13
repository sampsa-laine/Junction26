# 2D CAD Topology Optimization Arena

Environment for simulating, optimizing, and evaluating 2D structural CAD components against theoretical topology optimization baselines.

## How to Run

### Command Line Benchmark
Runs the 4-way comparison arena in the terminal:
```shell
uv run junction26
```

### Interactive Web Dashboard
Launches the Streamlit comparison arena with human designer canvas and CAD export:
```shell
uv run arena
```
Or directly:
```shell
uv run streamlit run src/agents/app.py
```

## The 4 Competitors

| Competitor | Methodology | Output | Role |
| :--- | :--- | :--- | :--- |
| **Classical SIMP** | Sigmund 88-line Optimality Criteria (OC) | Continuous density field | Ground truth theoretical upper bound on stiffness ($C^*$) |
| **LLM CAD Agent** | ReAct loop querying stress concentration | Parametric CAD primitives (holes, slots) | Visual/heuristic reasoning agent |
| **Dedicated Model** | Strain-energy guided hole placement | Parametric CAD cutouts | Algorithmic CAD baseline |
| **Human Designer** | Interactive Streamlit canvas | Manual user cuts | Human vs. AI challenge |

## Problem Presets

- `Cantilever Beam`: $60 \times 30$ rectangle, fixed left edge, downward corner load.
- `MBB Beam`: $60 \times 20$ span, pinned bottom-left, roller bottom-right, center load.
- `L-Bracket`: $40 \times 40$ domain with re-entrant corner cutout.
- `Suspension Arm`: $60 \times 30$ domain with 3 preserved bolt holes (keepouts).

## Module Layout (`src/agents/`)

- `problem.py`: Generalized domain, boundary supports, loads, keepout zones, and presets.
- `fea.py`: Vectorized 2D Q4 linear elastic FEA solver (<15ms per solve).
- `simp.py`: Generalized SIMP topology optimization solver.
- `csg.py`: 2D CAD geometry model using Shapely (boolean cutouts, rasterization, SVG).
- `metrics.py`: Evaluator (compliance, max stress, optimality gap, IoU, composite score).
- `heuristic_agent.py`: Dedicated model placing cutouts in low-strain regions.
- `llm_designer.py`: LLM ReAct agent with stress feedback.
- `export.py`: DXF (AutoCAD R12) and SVG export.
- `app.py`: Streamlit UI dashboard.
- `main.py`: CLI runner.
