# Break the Agent: 2D Structural CAD Lightweighting

**Today’s AI agents can write thousands of lines of code in seconds. Ask them to design a real physical part—and they fail basic physics.**

In aerospace and automotive engineering, removing excess weight without sacrificing strength is everything. Existing benchmarks test text or pixel grids you cannot machine—so we built a **real 2D CAD & Finite Element Analysis RL environment** (`<15ms` Q4 FEA) scored against **SIMP topology optimization ($C^*$)**, teaching AI to think in forces, geometry, and steel.

- **Machinable CAD**: Parametric 2D CSG cutouts (`circle`, `rect`, `slot`, `polygon`, fillets, mirrors) with DXF/SVG export
- **Millisecond Physics**: Vectorized Q4 FEA (Von Mises stress, strain energy, compliance)
- **Harbor + RL Suite**: Graded multi-load benchmarks with an isolated anti-cheat verifier

![Oracle CAD (0.94)](assets/suspension-rocker-multiload_oracle_geometry.png)
![Von Mises Stress](assets/suspension-rocker-multiload_oracle_stress.png)
![SIMP Optimum ($C^*$)](assets/suspension-rocker-multiload_simp_geometry.png)
![Strain Energy](assets/suspension-rocker-multiload_oracle_strain.png)
