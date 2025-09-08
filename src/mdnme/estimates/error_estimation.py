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
    is_nonmatching: bool = False,
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
    compute_diffusive_error(mdg)

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


def get_elements_to_refine(threshold: float = 0.10) -> None:
    """

    :param threshold:
    :return:
    """
    ...
