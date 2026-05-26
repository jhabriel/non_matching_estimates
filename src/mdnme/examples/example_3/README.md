# Example 3 — Small Features: 3D Fracture Network with Thin Features (Flow Benchmark 3D Case 3)

Single matching vs. non-matching comparison for the 3D benchmark with small geometric
features (Berre et al., 2021). No exact solution is available; results report the
majorant and dimension-aggregated error indicators.

## Files

| File | Description |
|------|-------------|
| `main.py` | Runs one matching and one non-matching solve, exports `results_small_features.csv`, and builds the LaTeX table. |
| `build_latex_table.py` | Reads `results_small_features.csv` and produces the LaTeX table for the paper. Can be run standalone with `--csv`. |
| `geometry.py` | Geometry mixin: matching grids from `benchmark_3d_case_3` (PorePy built-in) or non-matching grids via nested gmsh refinement of `mesh30k.geo`. |
| `model.py` | Mixer class assembling all components into `SmallFeaturesModel`. |
| `boundary_conditions.py` | Boundary conditions for the small-features benchmark. |
| `grids/` | Pre-built `.geo` files (`mesh30k`, `mesh140k`, `mesh350k`, `mesh500k`) and `fracture_network.csv`. |
| `results_small_features.csv` | Output of `main.py` (committed for reference). |
