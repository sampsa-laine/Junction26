# 2D Structural CAD Design Task: Cantilever Lightweight Bracket

## Objective
Engineer a lightweight, manufacturable 2D parametric CAD component saved at `/app/design.json`.
Starting from the stock plate in `/app/problem.json`, apply Constructive Solid Geometry (CSG) cutouts so the component meets the target mass budget while **minimizing structural compliance** (maximizing stiffness) across all load cases.

## Problem Specification & Engineering Constraints
- **Domain Bounding Box**: `60.0 x 30.0` (`60 x 30` Q4 finite element mesh)
- **Target Volume Fraction**: `<= 0.450` (measured relative to `60 x 30` bounding box)
- **Maximum von Mises Stress Limit**: `<= 1.400`
- **Minimum Cutout Feature Radius / Half-Width**: `>= 1.50`
- **Minimum Solid Wall Thickness**: `>= 1.50` (avoid paper-thin slivers or fragile bridges)
- **Maximum Parametric Cutouts**: `<= 20` actions
- **External Load Cases**:
  - **Load Case 1**: `(x=57.0, y=4.0, fx=+0.00, fy=-1.00)`
- **Keep-Out Regions** (`SOLID` bosses cannot be cut; `VOID` zones are pre-cut):
  - `SOLID` rect: params `(0.0, 0.0, 2.5, 30.0)`
  - `SOLID` circle: params `(57.0, 4.0, 3.0)`
- **Uncut Stock Baseline**: `volume_fraction = 1.0000`, `total_compliance = 32.99` (`case_compliances = [32.99]`), `max_von_mises_stress = 0.5796`

### Engineering Note
Primary bending tension/compression chords run along the top and bottom edges from the fixed left wall (x=0) down to the tip load boss at (57, 4), while the upper-right exterior corner and central web shear pockets carry minimal strain energy.

---

## Interactive Workbench Tool (`cad-sim`)

**Use only `cad-sim` CLI commands** to inspect the problem, apply CSG cutouts to `/app/design.json`, and evaluate FEA feedback (`<15ms` per call). **Do not write or run Python optimization/search scripts or import `cad_env` directly.**

```bash
# Inspect full specification & boundary conditions
cad-sim info

# Evaluate FEA compliance, stress hotspots, low-strain zones, and ASCII stress heatmap
cad-sim status --ascii

# Apply parametric CSG cutouts (automatically updates /app/design.json and runs FEA)
cad-sim cut-circle --x 30.0 --y 15.0 --r 4.5
cad-sim cut-slot --x1 18.0 --y1 12.0 --x2 38.0 --y2 12.0 --r 3.0
cad-sim cut-rect --x 20.0 --y 8.0 --w 12.0 --h 6.0 --fillet 2.0
cad-sim cut-polygon --points "25,28 55,28 55,14" --fillet 1.8

# Undo last cut or reset to uncut stock plate
cad-sim undo
cad-sim reset

# Optional: export CAD vector geometry
cad-sim export --svg /app/design.svg --dxf /app/design.dxf
```

### Workflow Tips
- **Use physics-guided cuts**: Inspect `cad-sim status --ascii` to see high-stress load paths (`9`/`#`), low-strain pockets (`1`/`2`), and `Low-Strain Zones`, then place/adjust cutouts intentionally.
- **Batch multiple cuts in one shell command when helpful**: You can chain planned cuts (e.g., `cad-sim cut-polygon ... && cad-sim cut-circle ... && cad-sim status --ascii`) to reduce round-trips.
- **Stop once satisfied**: Once `/app/design.json` meets the target volume fraction, stress limit, connectivity, and wall-thickness checks with strong stiffness, finish.

---

## Scoring & Verification (`/logs/verifier/reward.txt`)
Your final `/app/design.json` is graded on a **continuous scale from `0.0` to `1.0`**:
- **Hard Validity Checks (`0.0` if violated)**:
  1. `/app/design.json` must exist, parse cleanly, remove at least `2%` area, use `<= 20` cutouts, and obey `min_feature_radius >= 1.50`.
  2. A single connected solid structure must join the supports to **all** load points without floating solid islands or severed load paths.
- **Continuous Partial Credit (`0.0` to `1.0`)**:
  - Combines **stiffness efficiency** relative to the theoretical topological optimum ($C^*$), **volume budget satisfaction** (`1.0` when `volume_fraction <= 0.450 + 0.005`, smoothly penalized if over budget), **von Mises stress compliance** (`<= 1.400`), and **minimum wall thickness** (`>= 1.50`).
