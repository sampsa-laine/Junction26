## The Gap

Left to themselves, frontier models punch greedy holes wherever stress looks low—leaving behind fragile slivers, sharp stress fractures, or severed load paths. Our reference **Oracle** (`oracle_solver.py`) pairs **SIMP topology optimization ($C^*$)** with parametric CSG fitting to achieve a near-optimal strength-to-weight ratio, while **Do-Nothing** (uncut stock) and **Stop-Early** baselines earn zero credit.

### Why Coding Agents Break on Physical Design
- **No Spatial Lookahead**: Early cutouts without truss planning create weak hinges or disconnected islands.
- **Coupled Physics**: Every cut simultaneously shifts **volume**, **stiffness (compliance)**, and **peak Von Mises stress**.
- **Multi-Load Tradeoffs**: Optimizing for a single load case fails when a second independent load case is applied.
