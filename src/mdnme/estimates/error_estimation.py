from typing import Callable

import numpy as np
import porepy as pp

from mdnme.estimates.diffusive_error import compute_diffusive_error
from mdnme.estimates.flux_extension import extend_fv_fluxes
from mdnme.estimates.pressure_reconstruction import reconstruct_pressure
from mdnme.estimates.residual_error import compute_residual_error
from mdnme.utils.grid_rotation import assign_canonical_rotations


SpatialFunction = Callable[..., np.ndarray]


def estimate_errors(
    mdg: pp.MixedDimensionalGrid,
    pressure_reconstruction_method: str = "keilegavlen_p1",
    sources: list[SpatialFunction] | list[float] | None = None,
    quadrature_degree_for_residual_error: list[int] | None = None,
    non_matching_nested: bool = False,
) -> None:
    """Estimate local errors and save them in data dictionaries.

    Parameters:
        mdg: pp.MixedDimensionalGrid
            Mixed-dimensional grid for the given problem.
        pressure_reconstruction_method: str
            Name of the pressure reconstruction method to be employed,
            either `patchwise_p1` or `keilegavlen_p1`. If not given, the former is
            chosen by default.
        sources: List of callables or list of float, optional
            Exact source terms in each subdomain. Can be given as a list of spatial
            functions (i.e., depending on `x`, `y`, (and `z` if 3D)) or as a list of
            floats. If not given, we assume zero sources in each subdomain.
        quadrature_degree_for_residual_error: List of integers, optional
            Degree of quadrature rule to be used for the numerical integration of the
            residual error. To avoid introducing quadrature errors, the degree must
            be sufficiently high so that the sources can be integrated exactly. If not
            given, we employ 4.

    Note:
        On subdomains and interfaces, diffusive errors are stored in
        data["estimates"]["diffusive_error"]. On subdomains, residual errors
        are stored in data["estimates"]["residual_error"].

        We assume that all quantities of interest (i.e., pressures and fluxes) were
        reconstructed and are available in the data dictionaries.

    """

    # Perform extension of finite volume fluxes using RT0 basis functions
    extend_fv_fluxes(mdg)

    # Reconstruct cell-centered pressures
    reconstruct_pressure(mdg, method=pressure_reconstruction_method)

    # Error computation

    # Diffusive error
    compute_diffusive_error(mdg, non_matching_nested)

    # Residual error
    if sources is None:
        sources = [0.0 for _ in mdg.subdomains()]

    if quadrature_degree_for_residual_error is None:
        quadrature_degree_for_residual_error = [4 for _ in mdg.subdomains()]

    compute_residual_error(
        mdg,
        sources,
        quadrature_degree_for_residual_error,
    )


def get_majorant(mdg: pp.MixedDimensionalGrid) -> float:
    """
    Compute global upper bound.

    Parameters:
        mdg: pp.MixedDimensionalGrid
            Mixed-dimensional grid for the given problem.

    Returns:
        Global upper bound for the whole mixed-dimensional grid, a.k.a., the majorant.

    """
    sd_diffusive = 0.0
    sd_residual = 0.0
    intf_diffusive = 0.0

    # Errors associated to subdomains
    for sd, d in mdg.subdomains(return_data=True):
        if sd.dim != 0:
            sd_diffusive += d["estimates"]["diffusive_error"].sum()
            sd_residual += d["estimates"]["residual_error"].sum()

    # Errors associated to interfaces
    for intf, d in mdg.interfaces(return_data=True):
        intf_diffusive += d["estimates"]["diffusive_error"].sum()

    # Obtaining the majorant
    majorant = (sd_diffusive + intf_diffusive) ** 0.5 + sd_residual**0.5

    return majorant


def compute_sd_and_intf_errors_of_equal_dim(mdg: pp.MixedDimensionalGrid) -> dict:
    """
    Compute subdomain and interface errors of equal dimensionality.

    Parameters:
        mdg : pp.MixedDimensionalGrid

    Returns:
        Data dictionary with fields `subdomain_error` and `interface_error` and
        subfields d \in [0, 1, 2, 3], depending on the dimensionality of the
        mixed-dimensional grid.

    Note:
        We assume that the error indicators where already computed and are available
        in data["estimates"]["error_indicators"].

    """
    d = {}
    d['subdomain_error'] = {}
    d['interface_error'] = {}

    # Obtain max and min subdomain dim
    sd_dims = np.asarray([sd.dim for sd in mdg.subdomains()])
    min_sd_dims = np.min(sd_dims)
    max_sd_dims = np.max(sd_dims)
    dims_sd = np.arange(min_sd_dims, max_sd_dims+1)

    intf_dims = np.asarray([intf.dim for intf in mdg.interfaces()])
    min_intf_dims = np.min(intf_dims)
    max_intf_dims = np.max(intf_dims)
    dims_intf = np.arange(min_intf_dims, max_intf_dims+1)

    # Loop over the mixed-dimensional grid and calculate errors
    for dim in dims_sd:
        cum_error = 0
        for sd, data in mdg.subdomains(dim=dim, return_data=True):
            cum_error += data["estimates"]["error_indicator"].sum()
        d['subdomain_error'][dim] = np.sqrt(cum_error)

    for dim in dims_intf:
        cum_error = 0
        for intf, data in mdg.interfaces(dim=dim, return_data=True):
            cum_error += data["estimates"]["error_indicator"].sum()
        d['interface_error'][dim] = np.sqrt(cum_error)

    return d

def compute_error_indicators(mdg: pp.MixedDimensionalGrid) -> None:
    """
    Compute error indicators (i.e., to be used in the AMR process)

    Parameters:
        mdg: pp.MixedDimensionalGrid
            Mixed-dimensional grid for the given problem.

    Note:
        The data dictionary of each subdomain and interface of the
        mixed-dimensional grid will be updated with the field
        data["estimates"]["error_indicator"]

    """

    # Compute subdomain errors
    for sd, d in mdg.subdomains(return_data=True):
        # Create key if it does not exist
        if d["estimates"].get("error_indicator") is None:
            d["estimates"]["error_indicator"] = {}
        # Compute subdomain local error indicator
        d["estimates"]["error_indicator"] = (
            d["estimates"]["diffusive_error"] + d["estimates"]["residual_error"]
        ) ** 0.5

    # Compute interface errors
    for intf, d in mdg.interfaces(return_data=True):
        # Create key if it does not exist
        if d["estimates"].get("error_indicator") is None:
            d["estimates"]["error_indicator"] = {}
        # Compute subdomain local error indicator
        d["estimates"]["error_indicator"] = d["estimates"]["diffusive_error"]


def transfer_errors_iterate_solutions(mdg: pp.MixedDimensionalGrid) -> None:
    def transfer(data: dict, error_type: str):
        if error_type in data["estimates"]:
            data[pp.ITERATE_SOLUTIONS][error_type] = {}
            data[pp.ITERATE_SOLUTIONS][error_type][0] = data["estimates"][error_type]

            data[pp.TIME_STEP_SOLUTIONS][error_type] = {}
            data[pp.TIME_STEP_SOLUTIONS][error_type][0] = data["estimates"][error_type]

        else:
            raise ValueError("Estimates must be computed first.")

    # Transfer errors from subdomains
    for sd, d in mdg.subdomains(return_data=True):
        for error in ["diffusive_error", "residual_error", "error_indicator"]:
            transfer(d, error)

    # Transfer error from interfaces
    for _, d in mdg.interfaces(return_data=True):
        for error in ["diffusive_error", "error_indicator"]:
            transfer(d, error)