[![DOI](https://zenodo.org/badge/900789549.svg)](https://doi.org/10.5281/zenodo.17844409)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)


# `mdnme`: A posteriori error estimates for mixed-dimensional Darcy flow using non-matching grids

<p align="center">
  <img src="./mdnme_logo.png" alt="Repository Logo" width="400" height="300">
</p>

**mdnme** is a Python package for a posteriori error analysis of finite-volume approximations to the mixed-dimensional Darcy flow problem, supporting both matching and non-matching grids. The package is built on top of [PorePy](https://github.com/pmgbergen/porepy) and uses [quadpy](https://github.com/sigma-py/quadpy) for numerical integration.

## Installation

Please refer to the [installation instructions](Install.md).

## Package structure

```
src/mdnme/
├── estimates/          # A posteriori error estimation pipeline
│   ├── error_estimation.py         # Main entry point: estimate_errors(), get_majorant()
│   ├── diffusive_error.py          # Diffusive error indicators
│   ├── residual_error.py           # Residual error indicators
│   ├── gm_error.py                 # Grid-mismatch indicators (non-matching only)
│   ├── flux_extension.py           # RT0 flux extension
│   └── pressure_reconstruction.py  # Pressure post-processing
├── models/             # PorePy model components
│   ├── varela_jnum_2d/ # 2D manufactured-solution model (Varela et al., 2023)
│   └── varela_jnum_3d/ # 3D manufactured-solution model with fracture
└── examples/           # Reproducible paper examples
    ├── example_1/      # 3D single fracture with exact solution (convergence study)
    ├── example_2/      # 3D regular fracture network benchmark (Geiger case)
    └── example_3/      # 3D fracture network with small features benchmark
```

## Reproducing the paper examples

Each example in `src/mdnme/examples/` is self-contained. Run `main.py` from any location — outputs are always written to the example folder itself.

### Example 1 — Convergence study with exact solution
```bash
python src/mdnme/examples/example_1/main.py
```
Runs matching and non-matching solves at four mesh sizes, computes true errors and effectivity indices, and produces `results_raw.csv` and the consolidated LaTeX table. The guided subdomain halving experiment can be reproduced with:
```bash
python src/mdnme/examples/example_1/guided_halving.py
```

### Example 2 — Geiger 3D regular fracture network
```bash
python src/mdnme/examples/example_2/main.py
```
Runs matching and non-matching solves at three refinement levels (coarse/intermediate/fine) and produces `results_geiger3d.csv` and the LaTeX table. A parallel version (one process per refinement level) is also available:
```bash
python src/mdnme/examples/example_2/main_parallel.py
```

### Example 3 — 3D fracture network with small features
```bash
python src/mdnme/examples/example_3/main.py
```
Runs one matching and one non-matching solve and produces `results_small_features.csv` and the LaTeX table.

Results can be visualised by opening the generated `.vtu` files in [Paraview](https://www.paraview.org).

## Citing

If you use **mdnme** in your research, please cite:

> Varela, J., Schaerer, C. E., Keilegavlen, E., & Berre, I. (2025).
> *A posteriori error estimates for mixed-dimensional Darcy flow using non-matching grids.*
> arXiv preprint [arXiv:2512.09087](https://arxiv.org/abs/2512.09087).

## Issues

For feature requests, troubleshooting, and bug reports, please open an [issue](https://github.com/jhabriel/non_matching_estimates/issues).

## License

See [LICENSE](./LICENSE).

## Funding

The development of this software has received funding from CONACYT through project PRIA01-8 and from the Polytechnic University Taiwan-Paraguay.
