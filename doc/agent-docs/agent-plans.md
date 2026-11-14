# 2D CAD Simulation Environment: Architecture & Roadmap

Short overview of decisions and plans for building a 2D CAD simulation and topology optimization training environment within the 48-hour hackathon.

## Core Decisions

| Decision Area | Chosen Approach | Alternative Evaluated | Why Chosen |
| :--- | :--- | :--- | :--- |
| **Problem Definition** | Generalized 2D domain (custom BCs, loads, keep-outs, presets) | Hardcoded single problem (e.g. only cantilever) | Supports arbitrary 2D engineering components and custom user problems |
| **Geometry** | CSG with Shapely rasterized to grid | Density pixels only, or B-Rep splines | Real CAD operations (cutouts, holes) without meshing crashes |
| **FEA Engine** | Vectorized 2D Q4 quad FEA (SciPy sparse) | Unstructured triangular mesh (`scikit-fem`) | Sub-15ms solve time, zero meshing failures |
| **Ground Truth** | Vectorized SIMP TopOpt (Sigmund 88-line) | Level Set Method | Fast theoretical minimum compliance benchmark |
| **4-Way Comparison** | SIMP vs. LLM Agent vs. Dedicated Model vs. Human | Single agent only | Demonstrates LLM reasoning, benchmark optimality gap, and gamified human challenge |
| **Demo UI** | Streamlit | Gradio, Pygame | Side-by-side visualization, interactive human canvas, live leaderboard |

## The 4-Way Comparison Arena

1. **Classical TopOpt (SIMP)**: Ground-truth theoretical lower bound on compliance ($C^*$).
2. **LLM Designer Agent**: Iterative ReAct loop receiving stress hotspots and issuing CAD cut commands.
3. **Dedicated Model**: Stress-guided heuristic optimizer (low-strain removal) or parametric policy.
4. **Human Designer**: Interactive UI canvas where judges/users place holes to compete against AI and TopOpt.

## Generalized Problem Specification

```python
DesignProblem(
    domain_size=(width, height),
    grid_resolution=(nx, ny),
    supports=[Support(x, y, fixed_x=True, fixed_y=True)],
    loads=[Load(x, y, fx=0.0, fy=-1.0)],
    keepouts=[CircleKeepOut(x, y, r), RectKeepOut(...)],
    target_volume_fraction=0.40,
)
```
Presets provided: Cantilever, MBB Beam, L-bracket, Suspension Rocker, plus a Custom Problem Builder.

## Progression Paths

1. **Path 1 (Core Physics & SIMP Benchmark - Day 1 AM)**:
   - Generalized 2D Q4 FEA solver (`scipy.sparse`).
   - Generalized SIMP TopOpt solver for arbitrary loads & boundary conditions.
   - Unit tests on compliance convergence.

2. **Path 2 (CAD Geometry & Comparison Engine - Day 1 PM)**:
   - Shapely CSG engine (plate, cutouts, keep-outs, rasterizer).
   - Dedicated Model / Heuristic baseline (`src/agents/heuristic_agent.py`).
   - Unified evaluation & metrics engine (compliance, max stress, optimality gap).

3. **Path 3 (LLM ReAct Agent & Human Interface - Day 2 AM)**:
   - LLM design agent loop (`src/agents/llm_cad_agent.py`) with stress feedback and tool use.
   - Interactive Streamlit dashboard with 4-way leaderboard.

4. **Path 4 (Polish & Manufacturing Export - Day 2 PM)**:
   - DXF / SVG export.
   - Live demo rehearsal and documentation.

## 48-Hour Milestone Schedule

| Window | Focus | Deliverables |
| :--- | :--- | :--- |
| **H00 - H10** | Physics & Baseline | Vectorized Q4 FEA + Sigmund SIMP solver + tests |
| **H10 - H20** | Geometry & Env | Shapely CSG + `TopOptCADEnv` (Gymnasium) |
| **H20 - H32** | Agents | Heuristic agent + LLM ReAct agent + RL agent in `src/agents/` |
| **H32 - H42** | UI & Demo | Streamlit dashboard + live benchmark metrics |
| **H42 - H48** | Polish | DXF export + docs + pitch rehearsals |
