# 2D Structural CAD Design Task: Stress-Constrained L-Bracket

## Objective
Engineer a lightweight, manufacturable 2D parametric CAD component saved at `/app/design.json`.
Starting from the stock plate in `/app/problem.json`, apply Constructive Solid Geometry (CSG) cutouts so the component meets the target mass budget while **minimizing structural compliance** (maximizing stiffness) across all load cases.

## Problem Specification & Engineering Constraints
- **Domain Bounding Box**: `50.0 x 50.0` (`50 x 50` Q4 finite element mesh)
- **Target Volume Fraction**: `<= 0.360` (measured relative to `50 x 50` bounding box)
- **Maximum von Mises Stress Limit**: `<= 1.850`
- **Minimum Cutout Feature Radius / Half-Width**: `>= 1.50`
- **Minimum Solid Wall Thickness**: `>= 1.60` (avoid paper-thin slivers or fragile bridges)
- **Maximum Parametric Cutouts**: `<= 22` actions
- **External Load Cases**:
  - **Load Case 1**: `(x=46.0, y=10.0, fx=+0.00, fy=-1.00)`
- **Keep-Out Regions** (`SOLID` bosses cannot be cut; `VOID` zones are pre-cut):
  - `VOID` rect: params `(20.0, 20.0, 30.0, 30.0)`
  - `SOLID` rect: params `(0.0, 47.5, 20.0, 2.5)`
  - `SOLID` circle: params `(46.0, 10.0, 3.0)`

### Engineering Note
The void keep-out in [20, 50] x [20, 50] creates a sharp 90-degree re-entrant inner corner at (20, 20). Placing sharp cutouts or thinning the material near (20, 20) causes severe von Mises stress spikes above the allowable stress limit. Use smooth/filleted pockets away from high-stress load paths.

---

## Interactive Workbench Tools

### 1. Command-Line Tool (`cad-sim`)
Run `cad-sim` from anywhere to inspect the problem, apply CSG cutouts to `/app/design.json`, and run `<15ms` Q4 Finite Element Analysis (FEA):

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

### 2. Python Simulation API (`cad_env`)
You may also write and execute Python optimization/search scripts directly against the pre-installed `cad_env` package (`/opt/cad_env`):

```python
from cad_env import CADModel, DesignProblem, FEASolver, inspect_design

problem = DesignProblem.load_json("/app/problem.json")
solver = FEASolver(problem)
model = CADModel(problem)

model.cut_polygon([(25.0, 28.0), (56.0, 28.0), (56.0, 14.0)], fillet_radius=1.8)
model.cut_circle(22.0, 14.0, radius=4.0)

report = inspect_design(problem, model)
print(
    report["volume_fraction"],
    report["total_compliance"],
    report["max_von_mises_stress"],
)

# Always save your final CAD model to /app/design.json
model.save_json("/app/design.json")
```

---

## Scoring & Verification (`/logs/verifier/reward.txt`)
Your final `/app/design.json` is graded on a **continuous scale from `0.0` to `1.0`**:
- **Hard Validity Checks (`0.0` if violated)**:
  1. `/app/design.json` must exist, parse cleanly, remove at least `2%` area, use `<= 22` cutouts, and obey `min_feature_radius >= 1.50`.
  2. A single connected solid structure must join the supports to **all** load points without floating solid islands or severed load paths.
- **Continuous Partial Credit (`0.0` to `1.0`)**:
  - Combines **stiffness efficiency** relative to the theoretical topological optimum ($C^*$), **volume budget satisfaction** (`1.0` when `volume_fraction <= 0.360 + 0.005`, smoothly penalized if over budget), **von Mises stress compliance** (`<= 1.850`), and **minimum wall thickness** (`>= 1.60`).
