# Project Overview

2D CAD simulation and benchmark environment for training AI agents to design structural components closer to theoretical topological optimums.

## Objectives
- Provide a fast (<15ms) 2D linear elastic FEA simulation environment.
- Support authentic 2D CAD operations (CSG primitives, cutouts, keep-out zones).
- Benchmark AI-generated designs against theoretical SIMP topology optimization limits.
- Offer interactive comparison and gamified demos (Human vs. AI vs. TopOpt).

## Key Components
- **Simulation**: Vectorized Q4 finite element analysis (NumPy / SciPy).
- **Optimization**: Ground truth Sigmund 88-line SIMP solver.
- **Environment**: Gymnasium-compatible `TopOptCADEnv`.
- **Agents**: Heuristic, LLM ReAct, and RL agents in `src/agents/`.
- **Interface**: Interactive Streamlit dashboard with DXF/SVG export.
