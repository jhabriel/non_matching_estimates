[![PyTest](https://github.com/jhabriel/non_matching_estimates/actions/workflows/pytest.yml/badge.svg)](https://github.com/jhabriel/non_matching_estimates/actions/workflows/pytest.yml)
[![Code Style & Typing](https://github.com/jhabriel/non_matching_estimates/actions/workflows/code_style_and_typing.yml/badge.svg)](https://github.com/jhabriel/non_matching_estimates/actions/workflows/code_style_and_typing.yml)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

# mdnme: Mixed-Dimensional Error Estimates on Non-Matching Grids

<p align="center">
  <img src="./mdnme_logo.png" alt="Repository Logo" width="400" height="300">
</p>

**mdnme** is a Python package created to perform a posteriori error analysis on mixed-dimensional domains grids. The package is build on top of [PorePy](https://github.com/pmgbergen/porepy).

## Installation

Please, refer to the [installation instructions](https://github.com/jhabriel/non_matching_estimates/Install.md).

## Reproducing the examples of the paper

Examples from the article "A posteriori error estimates for mixed-dimensional elliptic equations on non-matching grids" (todo: add arxiv link) can be reproduced by running `main.py` in each example of the `examples` directory.
Each file outputs `txt` files containing the results of the paper. You can also visualize the local error estimators by opening the generated `vtu` files in a proper software e.g., [Paraview](https://www.paraview.org).

## Citing

If you use **mdmne** in your research, we ask you to cite the following reference:

todo: Add arxiv reference.

## Problems, suggestions, improvements?
Create an [issue](https://github.com/jhabriel/non_matching_estimates).

## License
See [license md](./LICENSE.md).
