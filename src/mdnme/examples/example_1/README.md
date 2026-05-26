# Example 1 — Varela J-Num 3D: Single Fracture with Exact Solution

Convergence study for a 3D single-fracture problem with a manufactured solution,
allowing computation of true errors and effectivity indices.

## Files

| File | Description |
|------|-------------|
| `main.py` | Full convergence study: runs matching and non-matching solves at four mesh sizes, computes true errors and effectivity indices, exports `results_raw.csv`, and builds the LaTeX table. |
| `build_latex_table.py` | Reads `results_raw.csv` and produces the consolidated LaTeX sideways-table (matching + non-matching columns). Can be run standalone with `--csv`. |
| `guided_halving.py` | Guided subdomain halving experiment: bisects the subdomain with the larger indicator at each step, producing the 4-row LaTeX table in the paper. |
| `non_matching_fig_gen.py` | Generates visualisation figures for the non-matching grid configuration. |
| `gmsh_frac_file.msh` | Pre-generated mesh for the single-fracture geometry. |
| `results_raw.csv` | Output of `main.py` (committed for reference). |
