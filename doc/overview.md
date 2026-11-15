# Project Overview

2D CAD simulation and benchmark environment for evaluating AI coding agents on structural lightweighting against theoretical topological optimums.

## Objectives
- Provide a fast (`<15ms`) 2D linear elastic Q4 FEA simulation environment.
- Support authentic 2D parametric CAD operations (CSG cutouts, fillets, mirrors, keep-out zones).
- Benchmark AI-generated designs in isolated Harbor v1.4 containers against SIMP topology optimization ($C^*$).
- Offer interactive comparison, FEA inspection, and `0..N` cutout replay in the Streamlit Arena.

## Key Components
- **Simulation & CSG**: Vectorized Q4 FEA (`fea.py`) and Shapely CSG `CADModel` (`csg.py`).
- **Optimization & Oracle**: Multi-load Sigmund SIMP solver (`simp.py`) and parametric SIMP-to-CAD oracle (`oracle_solver.py`).
- **Harbor Benchmark**: `cad-sim` CLI / `cad_env` sandbox, continuous `[0, 1]` verifier (`verifier.py`), and 4 tasks in `tasks/`.
- **Interface**: Interactive Streamlit Arena & Inspector (`uv run arena`) with SVG/DXF/JSON export.
