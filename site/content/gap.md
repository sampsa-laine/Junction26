## The Gap

Left to themselves, frontier models punch greedy holes wherever stress looks low—leaving behind fragile slivers, sharp stress fractures, or severed load paths. Our **reference implementation** pairs **topological optimization ($C^*$)** with parametric shape fitting to achieve a near-optimal strength-to-weight ratio, while **Do-Nothing** (uncut stock) and **Stop-Early** baselines earn zero credit.

### How Designs Are Scored & Compared
- **Theoretical Optimum ($C^*$)**: Each design's stiffness (compliance under load) is measured against the continuous **topological optimization** lower bound ($C^*$) for that geometry and load case.
- **Weight & Stress Budgets**: Passing designs must meet the target material volume (`<= 35%–40%` of the solid plate) and peak stress limit without fragile slivers or disconnected parts.
- **Reference vs. Baselines**: Frontier agents are compared directly against our **Reference Implementation** (parametric cutouts fitted to the $C^*$ optimum) and the **Do-Nothing Baseline** (uncut stock, scoring `0.0000`).

### Why Coding Agents Break on Physical Design
- **No Spatial Lookahead**: Early cutouts without truss planning create weak hinges or disconnected islands.
- **Coupled Physics**: Every cut simultaneously shifts **volume**, **stiffness (compliance)**, and **peak Von Mises stress**.
- **Multi-Load Tradeoffs**: Optimizing for a single load case fails when a second independent load case is applied.
