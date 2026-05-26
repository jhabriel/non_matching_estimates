# Example 2 — Geiger 3D: Regular Fracture Network (Flow Benchmark 3D Case 2)

Convergence study for the 3D regular fracture network benchmark (Berre et al., 2021).
No exact solution is available; results report the majorant and dimension-aggregated
error indicators for three refinement levels.

## Files

| File | Description |
|------|-------------|
| `main.py` | Sequential convergence study: runs matching and non-matching solves at three refinement levels (coarse/intermediate/fine), exports `results_geiger3d.csv`, and builds the LaTeX table. |
| `main_parallel.py` | Same study as `main.py` but runs the three refinement levels concurrently via `ProcessPoolExecutor` (max_workers=3). |
| `build_latex_table.py` | Reads `results_geiger3d.csv` and produces the LaTeX table for the paper. Can be run standalone with `--csv`. |
| `flow_benchmark_3d_case_2.py` | Defines the benchmark geometry, boundary conditions, and permeability specification following Section 5.2 of Berre et al. (2021). |
| `geometry.py` | Geometry mixin: loads matching grids from pre-built `.geo` files or generates non-matching grids via one-level nested gmsh refinement. |
| `model.py` | Mixer class assembling all components into `Geiger3dModel`. |
| `boundary_conditions.py` | Modified boundary conditions for the benchmark (inlet flux + outlet pressure). |
| `md_grids/` | Pre-built `.geo` and `.msh` files for the three refinement levels, plus `fracture_network.csv`. |
| `results_geiger3d.csv` | Output of `main.py` (committed for reference). |
