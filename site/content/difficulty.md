## Meaningful Difficulty

Every failure mode comes from genuine engineering decisions—not hidden rules or broken tools. Agents have full access to `cad-sim` / `cad_env` with `<15ms` FEA feedback, stress/strain telemetry, volume fraction, and wall-thickness checks at every step.

| Task | Grid | Target Vol | Core Engineering Challenge |
| :--- | :--- | :--- | :--- |
| `cantilever-lightweight` | `60x30` | `<= 0.45` | Truss triangulation between fixed wall (`x=0`) and tip boss `(57, 4)` |
| `l-bracket-stress` | `50x50` | `<= 0.36` | Relieving sharp `(20, 20)` re-entrant corner stress (`von Mises <= 1.85`) |
| `suspension-rocker-multiload` | `80x40` | `<= 0.40` | Balancing **3 mounting bosses** across **2 independent load cases** (bump + braking) |

### Observed Agent Failure Modes
- **Fragile Slivers & Severed Paths (`0.0000`)**: Greedy oversized cutouts break structural connectivity between supports and loads.
- **Sharp Stress Fractures**: Cutting near the L-bracket re-entrant corner spikes Von Mises stress above `1.85`.
- **Single-Load Overfitting**: On `suspension-rocker-multiload`, agents thin out struts needed for the secondary braking load case.

### Agent Sandbox Capabilities (`cad-sim` / `cad_env`)
- **Inspection**: `info` (problem bounds, keepouts, constraints) and `status` (`<15ms` Q4 FEA compliance, peak Von Mises stress, volume, connectivity, wall thickness).
- **Parametric Cutouts**: `cut-circle`, `cut-rect`, `cut-slot`, and `cut-polygon` (supporting optional fillets and mirror symmetry).
- **Iteration & Export**: `undo`, `reset` to uncut stock, and `export` to SVG / AutoCAD R12 DXF.
