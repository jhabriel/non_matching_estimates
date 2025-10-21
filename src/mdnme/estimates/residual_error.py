"""
This module contains functions that compute the local residual errors on subdomains.
We employ the local mass-conservative version of the residual error, see e.g.,
5.15 from [1].

Reference:
    [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. & Radu, F. (2023). A
      posteriori error estimates for hierarchical mixed-dimensional elliptic
      equations. Journal of Numerical Mathematics, 31(4), 247-280.
      https://doi.org/10.1515/jnma-2022-0038

"""
from __future__ import annotations

from typing import Callable

import numpy as np
import porepy as pp
import quadpy

import mdnme as amr

SpatialFunction = Callable[..., np.ndarray]


def compute_residual_error(
    mdg: pp.MixedDimensionalGrid,
    sources: list[float] | list[SpatialFunction],
    quadrature_degree: list[int],
) -> None:
    """Computes square of the residual error in al subdomain grids.

    Parameters:
        mdg: pp.MixedDimensionalGrid
            Mixed-dimensional grid.
        sources: List of floats or list of callable functions.
            Exact sources defined on each subdomain. Order should respect indexation
            in the mixed-dimensional grid.
        quadrature_degree: List of integers
            Degree of quadrature to perform the numerical integration.

    """
    for (sd, d), source, quad_deg in zip(
        mdg.subdomains(return_data=True), sources, quadrature_degree
    ):
        # Retrieve residual error and store in the dictionary
        d["estimates"]["residual_error"] = _residual_error(
            sd,
            d,
            source,
            quad_deg,
        )


def _residual_error(
    sd: pp.Grid,
    d: dict,
    external_sources: Callable[..., np.ndarray] | float,
    quad_deg: int,
) -> np.ndarray:
    """Computes the square of the residual error for every element of the grid.

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        d: dict
            Data dictionary associated to the grid
        external_sources: Callable or float
            External source. Either a callable function or a float.
        quad_deg: int
            Degree of quadrature used in the numerical integration scheme.

    Returns:
        Residual error (squared) for all cells of the subdomain grid. The size of the
        numpy array is `sd.num_cells`.

    """
    # Retrieve RT0 fluxes and compute divergence of the flux
    recon_u = d["estimates"]["recon_sd_flux"]
    if recon_u is not None:
        u = amr.utils.poly2col(recon_u)
        div_u: np.ndarray = sd.dim * u[0]

    # Retrieve contribution from higher-dimensional neighboring interfaces
    intf_jump: np.ndarray = d["estimates"]["sources_from_intf"]

    # Declare integration method
    if sd.dim == 1:
        method = quadpy.c1.newton_cotes_closed(quad_deg)
    elif sd.dim == 2:
        method = quadpy.t2.get_good_scheme(quad_deg)
    else:
        method = quadpy.t3.get_good_scheme(quad_deg)

    # Arrange elements in quadpy format
    if sd.dim in [1, 2, 3]:
        elements = amr.utils.get_quadpy_elements(sd)

    # Declare integrand
    def integrand(x: np.ndarray) -> np.ndarray:
        if not isinstance(external_sources, float):
            if sd.dim == 3:
                out = (external_sources(x[0], x[1], x[2]) - div_u + intf_jump) ** 2
            elif sd.dim == 2:
                out = (external_sources(x[0], x[1]) - div_u + intf_jump) ** 2
            else:
                out = (external_sources(x[0]) - div_u + intf_jump) ** 2
        else:
            out = ((external_sources - div_u + intf_jump) * np.ones_like(x[0])) ** 2
        return out

    # Perform numerical integration
    if sd.dim in [1, 2, 3]:
        integral = method.integrate(integrand, elements)
    else:
        integral = (external_sources + intf_jump.flatten()) ** 2

    return integral
