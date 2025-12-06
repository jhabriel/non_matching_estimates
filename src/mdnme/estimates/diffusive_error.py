"""
This module contains functions that compute the local diffusive errors on subdomains
and interfaces, see e.g., 5.7 from [1].

Reference:
    [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. & Radu, F. (2023). A
      posteriori error estimates for hierarchical mixed-dimensional elliptic
      equations. Journal of Numerical Mathematics, 31(4), 247-280.
      https://doi.org/10.1515/jnma-2022-0038

"""

from __future__ import annotations

from typing import Union

import numpy as np
import porepy as pp
import quadpy
import scipy.sparse as sps

import mdnme
from mdnme.estimates.helpers import is_nonmatching
from mdnme.utils.internal_boundary_grid import (
    InternalBoundaryGrid,
    InternalBoundaryLineGrid,
)
from mdnme.utils.primal_projections import (
    project_p1_1d_sz,
    prolong_to_transfer,
    scott_zhang_quasi_interpolant,
)
from mdnme.utils.transfer_grid import (
    TransferGrid,
    TransferLine,
    build_transfer_grid_nested,
    coarse_fine_or_build,
)


def compute_diffusive_error(
    mdg: pp.MixedDimensionalGrid,
    is_non_matching: bool,
) -> None:
    """Computes square of the diffusive flux error in all the mixed-dimensional grid.

    In each data dictionary, the square of the diffusive flux error will be stored
    in data["estimates"]["diffusive_error"].

    """
    # Loop through subdomains
    for sd, d in mdg.subdomains(return_data=True):
        # Handle the case of zero-dimensional subdomains
        if sd.dim == 0:
            d["estimates"]["diffusive_error"] = np.array([0.0])
            continue
        # Retrieve subdomain diffusive error
        d["estimates"]["diffusive_error"] = subdomain_diffusive_error(sd, d)

    # Loop through interfaces
    for intf, data_intf in mdg.interfaces(return_data=True):
        # Retrieve subdomain pair from interface
        sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
        data_high = mdg.subdomain_data(sd_high)
        data_low = mdg.subdomain_data(sd_low)
        # Retrieve interface diffusive error
        data_intf["estimates"]["diffusive_error"] = interface_diffusive_error(
            intf, data_intf, sd_high, data_high, sd_low, data_low, is_non_matching
        )


def subdomain_diffusive_error(sd: pp.Grid, d: dict) -> np.ndarray:
    """Computes the square of the subdomain diffusive errors.

    The square of the diffusive flux error is given locally for an element E by:

            || K_E^(-1/2) u_rec,E + K_E^(1/2) grad(p_rec,E) ||_E^2,

    where K_E is the effective tangential permeability, u_rec,E is the reconstructed
    velocity, and grad(p_rec,E) is the gradient of the reconstructed pressure.

    Raises:
        ValueError:
            - If the reconstructed pressure is not in the data dictionary.
            - If the reconstructed velocity is not in the data dictionary.
            - If the grid dimension is not 1, 2, or 3.

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        d: dict
            Data dictionary. We assume the keys `recon_sd_pressure` and
            `recon_sd_flux` are present in `d['estimates']`.

    Returns
        Square of the diffusive flux error for the  grid. The size of the array is
        sd.num_cells.

    """
    # Sanity checks
    if sd.dim not in [1, 2, 3]:
        raise ValueError("Error not defined for the given grid dimension.")
    if "recon_sd_pressure" not in d["estimates"]:
        raise ValueError("Pressure must be reconstructed first.")
    if "recon_sd_flux" not in d["estimates"]:
        raise ValueError("Fluxes must be extended first.")

    # Retrieve reconstructed pressure and extended fluxes coefficients
    recon_p = d["estimates"]["recon_sd_pressure"]
    recon_u = d["estimates"]["recon_sd_flux"]

    # Retrieve effective tangential permeability
    # TODO: Use full tensor in the calculation of diffusive errors
    perm_tensor = d[pp.PARAMETERS]["flow"]["second_order_tensor"].values
    k = np.reshape(perm_tensor[0][0], (sd.num_cells, 1))

    # Get QuadPy elements and declare integration method
    elements = mdnme.utils.get_quadpy_elements(sd)
    if sd.dim == 1:
        method = quadpy.c1.newton_cotes_closed(4)
    elif sd.dim == 2:
        method = quadpy.t2.get_good_scheme(4)
    else:
        method = quadpy.t3.get_good_scheme(4)

    # Obtain coefficients
    p = mdnme.utils.poly2col(recon_p)
    u = mdnme.utils.poly2col(recon_u)

    # Declare integrands and prepare for integration
    def integrand(x):
        # One-dimensional subdomains
        if sd.dim == 1:
            veloc_x = u[0] * x + u[1]

            gradp_x = p[0] * np.ones_like(x)

            int_x = (k ** (-0.5) * veloc_x + k**0.5 * gradp_x) ** 2

            return int_x

        # Two-dimensional subdomains
        elif sd.dim == 2:
            veloc_x = u[0] * x[0] + u[1]
            veloc_y = u[0] * x[1] + u[2]

            gradp_x = p[0] * np.ones_like(x[0])
            gradp_y = p[1] * np.ones_like(x[1])

            int_x = (k ** (-0.5) * veloc_x + k**0.5 * gradp_x) ** 2
            int_y = (k ** (-0.5) * veloc_y + k**0.5 * gradp_y) ** 2

            return int_x + int_y

        # Three-dimensional subdomains
        else:
            veloc_x = u[0] * x[0] + u[1]
            veloc_y = u[0] * x[1] + u[2]
            veloc_z = u[0] * x[2] + u[3]

            gradp_x = p[0] * np.ones_like(x[0])
            gradp_y = p[1] * np.ones_like(x[1])
            gradp_z = p[2] * np.ones_like(x[2])

            int_x = (k ** (-0.5) * veloc_x + k**0.5 * gradp_x) ** 2
            int_y = (k ** (-0.5) * veloc_y + k**0.5 * gradp_y) ** 2
            int_z = (k ** (-0.5) * veloc_z + k**0.5 * gradp_z) ** 2

            return int_x + int_y + int_z

    # Compute the integral
    diffusive_error = method.integrate(integrand, elements)

    return diffusive_error


def interface_diffusive_error(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
    is_non_matching: bool,
) -> np.ndarray:
    """Computes the square of the diffusive error on interfaces.

    The diffusive error on the interfaces is given locally for an interface cell E by:

        || k_E^(-1/2) lambda_E + k_E^(1/2) (pl_rec,E - tr ph_rec,E) ||_E^2,

    where k_E is the normal diffusivity, lambda_E is the normal velocity, pl_rec,E is
    the reconstructed lower-dimensional pressure, and tr ph_rec,E is the trace of the
    reconstructed higher-dimensional pressure.

    Raises:
        ValueError:
            - If the mortar grid dimension is not 0, 1, or 2.

    Parameters:
        intf: pp.MortarGrid
            Interface grid.
        data_intf: dict
            Interface data dictionary.
        sd_high: pp.Grid
            Higher-dimensional neighboring subdomain grid.
        data_high: dict
            Higher-dimensional neighboring data dictionary.
        sd_low: pp.Grid
            Lower-dimensional neighboring subdomain grid.
        data_low: dict
            Lower-dimensional neighboring data dictionary.

    Returns:
        Diffusive error (squared) for each element of the interface grid. The size of
        the np.ndarray is intf.num_cells.

    """
    # Check dimensionality of the mortar grid
    if intf.dim not in [0, 1, 2]:
        raise ValueError("Inconsistent mortar grid dimension. Expected 0, 1, or 2.")

    # Obtain diffusive error depending on the dimensionality of the grid
    if intf.dim == 0:
        diffusive_error = _interface_diffusive_error_0d(
            intf,
            data_intf,
            sd_high,
            data_high,
            sd_low,
            data_low,
        )
    elif intf.dim == 1:
        if not is_non_matching:
            diffusive_error = _interface_diffusive_error_1d(
                intf,
                data_intf,
                sd_high,
                data_high,
                sd_low,
                data_low,
            )
        else:
            diffusive_error = _interface_diffusive_error_1d_nonmatching(
                intf,
                data_intf,
                sd_high,
                data_high,
                sd_low,
                data_low,
            )
    else:
        if not is_non_matching:
            diffusive_error = _interface_diffusive_error_2d(
                intf,
                data_intf,
                sd_high,
                data_high,
                sd_low,
                data_low,
            )
        else:
            diffusive_error = _interface_diffusive_error_2d_nonmatching(
                intf,
                data_intf,
                sd_high,
                data_high,
                sd_low,
                data_low,
                tol=1e-8,
            )

    return diffusive_error


# Private functions
def _get_high_pressure_trace(
    sd_low: pp.Grid,
    sd_high: pp.Grid,
    data_sd_high: dict,
    frac_faces: np.ndarray,
    rotation_matrix: np.ndarray | None = None,
    dim_bool: np.ndarray | None = None,
) -> np.ndarray:
    """Obtains the coefficients of the P1 (projected) traces of the pressure.

    Raises:
        ValueError:
            - If the pressure has not been reconstructed.

    Parameters:
        sd_low: pp.Grid
            Lower-dimensional neighboring subdomain grid.
        sd_high: pp.Grid
            Higher-dimensional neighboring subdomain grid.
        data_sd_high: dict
            Higher-dimensional neighboring data dictionary.
        frac_faces: np.ndarray
            Indices of the higher-dimensional fracture faces.

    Returns:
        Coefficients of the higher-dimensional pressure trace.

    """

    def get_intf_lagrangian_coo(grid: Union[pp.Grid, mdnme.RotatedGrid]) -> np.ndarray:
        """Gets coordinates of the Lagrangian nodes of the internal higher-dim boundary.

        Parameters:
            grid: pp.Grid or amr.RotatedGrid:
                Higher-dimensional grid.

        Returns:
            Coordinates of the Lagrangian nodes.

        """
        # Get nodes of the fracture faces
        nodes_of_frac_faces = sps.find(sd_high.face_nodes.T[frac_faces])[1].reshape(
            frac_faces.size, sd_high.dim
        )

        # Obtain the coordinates of the nodes of the fracture faces
        lagran_coo = grid.nodes[:, nodes_of_frac_faces]

        return lagran_coo

    # 1) always use the *same* rotation used to build p_high
    gh_rot = mdnme.rotate_grid(sd_high)

    # 2) choose the target frame
    if rotation_matrix is None or dim_bool is None:
        gl_rot = mdnme.rotate_grid(sd_low)
        rotation_matrix = gl_rot.rotation_matrix
        dim_bool = gl_rot.dim_bool

    # Cells that correspond to frac_faces
    cells_of_frac_faces = sps.find(sd_high.cell_faces[frac_faces])[1]

    if "recon_sd_pressure" not in data_sd_high["estimates"]:
        raise ValueError("Pressure must be reconstructed first")
    p_high_full = data_sd_high["estimates"]["recon_sd_pressure"]
    p_high = p_high_full[cells_of_frac_faces]

    # 3) evaluate p_high in the *correct* (subdomain) frame
    point_coo_rot = get_intf_lagrangian_coo(gh_rot)
    point_val = mdnme.utils.evaluate_p1(p_high, point_coo_rot)

    # 4) rotate the same physical points into the target frame
    point_coo = get_intf_lagrangian_coo(sd_high)  # original 3D coords
    point_edge_coo_rot = np.empty_like(point_coo)
    for element in range(frac_faces.size):
        point_edge_coo_rot[:, element] = rotation_matrix @ point_coo[:, element]
    point_edge_coo_rot = point_edge_coo_rot[dim_bool]

    # 5) fit a P1 in the target frame
    trace_pressure = mdnme.utils.interpolate_p1(point_val, point_edge_coo_rot)

    # sanity check: in target frame, this P1 reproduces values at those points
    point_val_rot = mdnme.utils.evaluate_p1(trace_pressure, point_edge_coo_rot)
    np.testing.assert_allclose(point_val, point_val_rot, rtol=1e-9, atol=1e-8)

    return trace_pressure


def _get_low_pressure(data_low: dict, frac_cells: np.ndarray) -> np.ndarray:
    """Obtains the coefficients of the projected lower-dimensional pressure.

    Raises:
        ValueError
            - If the pressure has not been reconstructed.

    Parameters:
        data_low: dict
            Lower-dimensional data dictionary.
        frac_cells: np.ndarray
            Lower-dimensional fracture cells.

    Returns:
        Coefficients of the projected lower-dimensional pressure.

    """
    # Retrieve lower-dimensional reconstructed pressure coefficients
    if "recon_sd_pressure" in data_low["estimates"]:
        p_low = data_low["estimates"]["recon_sd_pressure"]
    else:
        raise ValueError("Pressure must be reconstructed first")
    p_low = p_low[frac_cells]

    return p_low


def _get_normal_velocity(intf: pp.MortarGrid, d: dict) -> np.ndarray:
    """Obtains the normal velocities for each interface cell.

    The normal velocities are the interface fluxes scaled by the interface cell measure.
    That is, area in 2D, length in 1D, and the unity in 0D.

    Raises:
        ValueError
            - If the mortar fluxes are not in the data dictionary

    Parameters:
        intf: pp.MortarGrid.
            Interface grid.
        d: dict
            Interface data dictionary.

    Returns:
        Normal velocities at the interfaces.

    """
    # Retrieve interface fluxes from the interface dictionary
    if "fv_intf_flux" in d["estimates"]:
        mortar_flux: np.ndarray = d["estimates"]["fv_intf_flux"]
    else:
        raise ValueError("Interface fluxes not found in the data dictionary")

    # Get hold of mortar grid and obtain the volumes of the mortar cells
    cell_volumes = intf.cell_volumes

    # Obtain the normal velocities and reshape into a column array
    normal_velocity = mortar_flux / cell_volumes
    normal_velocity = normal_velocity.reshape(mortar_flux.size, 1)

    return normal_velocity


# Interface errors
def _interface_diffusive_error_0d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
) -> np.ndarray:
    """Computes interface diffusive flux error on 0d interface grids.

    Raises:
        ValueError:
            - If the dimension of the mortar grid is different from zero.
            - If the reconstructed pressures are not found in the data dictionaries.
            - If the mortar flux is not found in the edge dictionary.

    Parameters:
        intf: pp.MortarGrid
            Interface grid.
        data_intf: dict
            Interface data dictionary.
        sd_high: pp.Grid
            Higher-dimensional neighboring subdomain grid.
        data_high: dict
            Higher-dimensional neighboring data dictionary.
        sd_low: pp.Grid
            Lower-dimensional neighboring subdomain grid.
        data_low: dict
            Lower-dimensional neighboring data dictionary.

    Returns:
         Diffusive error (squared) for each interface cell. The size of the array is
         intf.num_cells.

    """
    # Sanity checks
    if intf.dim != 0:
        raise ValueError("Expected zero-dimensional mortar grid.")
    if "recon_sd_pressure" not in data_high["estimates"]:
        raise ValueError("Pressure must be reconstructed first.")
    if "recon_sd_pressure" not in data_low["estimates"]:
        raise ValueError("Pressure must be reconstructed first.")
    if "fv_intf_flux" not in data_intf["estimates"]:
        raise ValueError("Mortar fluxes not found in the data dictionary")

    # Retrieve effective permeability
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    if isinstance(eff_perm, int) or isinstance(eff_perm, float):
        k = eff_perm * np.ones([intf.num_cells])
    else:
        k = eff_perm

    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
    frac_cells = sps.find(intf.secondary_to_mortar_avg())[1]

    # Rotate 1d-grid
    sd_rot = mdnme.rotate_grid(sd_high)

    # Obtain the trace of the pressure of the 1D grid
    cells_of_frac_faces = sps.find(sd_high.cell_faces[frac_faces])[1]
    p_1d = data_high["estimates"]["recon_sd_pressure"]
    p_1d = p_1d[cells_of_frac_faces]
    coo_frac_faces = sd_rot.face_centers[:, frac_faces].T
    coo_frac_faces = coo_frac_faces[np.newaxis, :, :]
    trace_p = mdnme.utils.evaluate_p1(p_1d, coo_frac_faces).flatten()

    # Obtain the pressure of the 0D grid
    p_0d = data_low["estimates"]["recon_sd_pressure"]
    p_0d = p_0d[frac_cells]

    # Pressure jump
    p_jump = p_0d - trace_p

    # Retrieve mortar solution
    mortar_flux = data_intf["estimates"]["fv_intf_flux"]
    normal_vel = mortar_flux / intf.cell_volumes

    # NOTE: We don't really need to use sidegrids in this case, since
    # the pressure in 0D domains is unique
    diffusive_error = (k ** (-0.5) * normal_vel + k**0.5 * p_jump) ** 2

    return diffusive_error


def _interface_diffusive_error_1d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
) -> np.ndarray:
    """Computes diffusive flux error (squared) for one-dimensional interface grids.

    Raises:
        ValueError
            - If the dimension of the mortar grid is different from 1.

    Parameters:
        intf: pp.MortarGrid
            Interface grid.
        data_intf: dict
            Interface data dictionary.
        sd_high: pp.Grid
            Higher-dimensional neighboring subdomain grid.
        data_high: dict
            Higher-dimensional neighboring data dictionary.
        sd_low: pp.Grid
            Lower-dimensional neighboring subdomain grid.
        data_low: dict
            Lower-dimensional neighboring data dictionary.

    Returns:
        Diffusive error (squared) for each cell of the interface grid. The size of
        the array is intf.num_cells.

    """

    def compute_sidegrid_error(side_tuple: tuple) -> np.ndarray:
        """Projects an interface quantity to a side grid and perform integration.

        Parameters:
            side_tuple: tuple
                Containing the side grids.

        Returns:
            Diffusive error (squared) for each element of the side grid.

        """
        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]

        # Obtain quadpy elements
        elements = mdnme.utils.get_quadpy_elements(sidegrid)

        # Project relevant quantities to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k

        # Declare integrand
        def integrand(x):
            coors = x[np.newaxis, :, :]  # add new axis, this is needed for 1D grids
            p_jump = mdnme.utils.evaluate_p1(deltap_side, coors)
            return (k_side ** (-0.5) * normalvel_side + k_side**0.5 * p_jump) ** 2

        # Compute integral
        diffusive_error_side = method.integrate(integrand, elements)

        return diffusive_error_side

    # Sanity check on mortar grid dimension
    if intf.dim != 1:
        raise ValueError("Expected one-dimensional interface grid.")

    # Retrieve effective normal permeability
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    if isinstance(eff_perm, int) or isinstance(eff_perm, float):
        k = eff_perm * np.ones([intf.num_cells, 1])
    else:
        k = eff_perm.reshape(intf.num_cells, 1)

    # BUG: Replacing interface grids (even in a equivalent grid) is causing issues
    # The problem is that the projector operators are wrong. Perhaps the problem
    # is that I'm trying to replace by an equivalent grid... yikes...
    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
    frac_cells = sps.find(intf.secondary_to_mortar_avg())[1]

    # Obtain the trace of the higher-dimensional pressure
    tracep_high = _get_high_pressure_trace(sd_low, sd_high, data_high, frac_faces)

    # Obtain the lower-dimensional pressure
    p_low = _get_low_pressure(data_low, frac_cells)

    # Now, we can work with the pressure difference
    deltap = p_low - tracep_high

    # Obtain normal velocities
    normal_vel = _get_normal_velocity(intf, data_intf)

    # Declare integration method
    method = quadpy.c1.newton_cotes_closed(4)

    # Retrieve side-grids tuples
    sides = intf.project_to_side_grids()

    # Compute the errors for each side grid
    diffusive = []
    for side in sides:
        diffusive.append(compute_sidegrid_error(side))

    # Concatenate into one numpy array
    diffusive_error = np.concatenate(diffusive)

    return diffusive_error


def _interface_diffusive_error_2d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
) -> np.ndarray:
    """Computes diffusive flux error (squared) for two-dimensional interface grids.

    Raises:
        ValueError
            - If the dimension of the interface grid is different from 2.

    Parameters:
        intf: pp.MortarGrid
            Interface grid.
        data_intf: dict
            Interface data dictionary.
        sd_high: pp.Grid
            Higher-dimensional neighboring subdomain grid.
        data_high: dict
            Higher-dimensional neighboring data dictionary.
        sd_low: pp.Grid
            Lower-dimensional neighboring subdomain grid.
        data_low: dict
            Lower-dimensional neighboring data dictionary.

    Returns:
        Diffusive error (squared) for each cell of the interface grid. The size of
        the array is intf.num_cells.

    """

    def compute_sidegrid_error(side_tuple: tuple) -> np.ndarray:
        """Projects an interface quantity to a side grid and perform integration.

        Parameters:
            side_tuple: tuple
                Containing the side grids.

        Returns:
            Diffusive error (squared) for each element of the side grid.

        """
        # Get projector and sidegrid object
        projector = side_tuple[0]
        sidegrid = side_tuple[1]

        # Obtain quadpy elements
        elements = mdnme.utils.get_quadpy_elements(sidegrid)

        # Project relevant quantities to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k

        # Declare integrand
        def integrand(x):
            p_jump = mdnme.utils.evaluate_p1(deltap_side, x)
            return (k_side ** (-0.5) * normalvel_side + k_side**0.5 * p_jump) ** 2

        # Compute integral
        diffusive_error_side = method.integrate(integrand, elements)

        return diffusive_error_side

    # Sanity check on dimensionality of mortar grid
    if intf.dim != 2:
        raise ValueError("Expected two-dimensional interface grid.")

    # Retrieve effective normal permeability
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    if isinstance(eff_perm, int) or isinstance(eff_perm, float):
        k = eff_perm * np.ones([intf.num_cells, 1])
    else:
        k = eff_perm.reshape(intf.num_cells, 1)

    # Face-cell map between higher- and lower-dimensional subdomains
    frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
    frac_cells = sps.find(intf.secondary_to_mortar_avg())[1]

    # Obtain the trace of the higher-dimensional pressure
    tracep_high = _get_high_pressure_trace(sd_low, sd_high, data_high, frac_faces)

    # Obtain the lower-dimensional pressure
    p_low = _get_low_pressure(data_low, frac_cells)

    # Now, we can work with the pressure difference
    deltap = p_low - tracep_high

    # Obtain normal velocities
    normal_vel = _get_normal_velocity(intf, data_intf)

    # Declare integration method
    method = quadpy.t2.get_good_scheme(4)

    # Retrieve side-grids tuples
    sides = intf.project_to_side_grids()

    # Compute the errors for each side grid
    diffusive = []
    for side in sides:
        diffusive.append(compute_sidegrid_error(side))

    # Concatenate into one numpy array
    diffusive_error = np.concatenate(diffusive)

    return diffusive_error


def _interface_diffusive_error_1d_nonmatching(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
    tol: float = 1e-8,
) -> np.ndarray:
    """Non-matching 1D interface diffusive error.

    Note:
        We compute the following norm:
                || k^{-1/2} λ + k^{1/2} (p_low − tr p_high) ||^2
        per mortar cell.

    Geometry: 2D high grid ↔ 1D fracture grid via 1D MortarGrid.

    The workflow mirrors `_interface_diffusive_error_2d_nonmatching`:

      1. Read effective permeability on the mortar and normal velocities.
      2. Read reconstructed P1 pressures on the low-dim (1D) fracture grid.
      3. Read high-dim pressure traces on the high edges that feed the interface.
      4. Build a *single* InternalBoundaryLineGrid per interface (contains both
         sides, per-side 1D IBGs).
      5. Loop over sides in `intf.project_to_side_grids()`:
         - pull the side enum,
         - grab the side IBG and parent-edge map,
         - map high trace P1 coeffs to IBG cells via the parent map,
         - transfer IBG→mortar-side and frac→mortar-side via TransferGrid
           + Scott–Zhang quasi-interpolation,
         - compute the jump (p_low - tr p_high) on the mortar side,
         - integrate (k^{-1/2} λ + k^{1/2} jump)^2 over each mortar-side cell
           using 1D quadpy,
         - scatter back to the global mortar ordering.

    Returns
    -------
    out_global : np.ndarray, shape (intf.num_cells,)
        Squared diffusive error per mortar cell.

    """

    # --- sanity ---
    if intf.dim != 1:
        raise ValueError("Expected one-dimensional interface grid.")

    # --- mortar-side scalars ---
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    k_mortar = (
        float(eff_perm) * np.ones((intf.num_cells, 1))
        if np.isscalar(eff_perm)
        else np.asarray(eff_perm, dtype=float).reshape(-1, 1)
    )

    normal_vel_mortar = _get_normal_velocity(intf, data_intf)  # (n_mortar, 1)

    # --- low-dim pressure (per-cell P1 on its own 1D grid) ---
    # shape: (n_frac_cells, n_p1_dofs=2)
    p_low_frac = data_low["estimates"]["recon_sd_pressure"]

    # --- edge-trace of high-dim pressure, in interface frame ---
    # primary_to_mortar_avg maps high *edges* to mortar cells for a 1D interface
    frac_edges = sps.find(intf.primary_to_mortar_avg())[1]
    # NOTE: we reuse the same helper name as in the 2D code; if you had a
    # dedicated 1D version, plug it here instead.
    p_trace_high = _get_high_pressure_trace(
        sd_low,
        sd_high,
        data_high,
        frac_edges,
    )  # (n_frac_edges, n_p1_dofs)
    # map: high edge id -> local index into frac_edges
    edge2pos = {int(e): i for i, e in enumerate(frac_edges)}

    # --- IBG: one per interface, with per-side IBG 1D grids ---
    ibg = InternalBoundaryLineGrid(intf, sd_high, tol=tol)

    # --- quadrature on 1D mortar sides ---
    method = quadpy.c1.newton_cotes_closed(4)

    # accumulator in global mortar ordering
    out_global = np.zeros(intf.num_cells)

    # Canonical rotation matrix
    rot_matrix, dim_bool, _ = mdnme.canonical_frame(intf)

    # loop sides in the mortar’s canonical order
    for P_msg, mg_side in intf.project_to_side_grids():

        # identify the side enum that owns this mortar side grid
        side_enum = next(k for k, v in intf.side_grids.items() if v is mg_side)

        # IBG side grid and its parent edges (one parent edge per IBG cell)
        ibg_side = ibg.ibg_side_grid(side_enum)
        parent_edges = ibg.parent_edge_of_cell(side_enum)  # (n_ibg_cells,)

        # (1) IBG-side tr(p_high): assign edge P1 coeffs to each IBG cell via parent map
        idx = np.fromiter(
            (edge2pos[int(e)] for e in parent_edges),
            dtype=int,
            count=parent_edges.size,
        )
        tr_hi_on_ibg = p_trace_high[idx, :]  # (n_ibg_cells, n_p1_dofs)

        # (2) Transfer IBG→mortar-side and frac→mortar-side

        # TODO: This is turning extremely hacky. What should be done?
        #  Well, need to construct, both the internal boundary grid and the
        #  transfer grid using physical coordinates (similar to what a side-grid
        #  or a fracture grid is constructed), avoiding the need of keeping track
        #  of which grid should be rotated and which should not. In other words,
        #  ALL grids are rotated ALWAYS. This is a little bit less efficient,
        #  but is worth it.

        # Internal boundary side grid -> Mortar side grid
        tracep_on_msg = project_p1_1d_sz(
            ibg_side,
            mg_side,
            tr_hi_on_ibg,
            tol=tol,
            rotation_matrix=rot_matrix,
            dim_bool=dim_bool,
            rotate_source=False,  # IBG is already given in rotated coordinates.
            rotate_target=True,  # Mortar side grid is in physical coordinates.
        )

        # Fracture grid -> Mortar Side grid
        fracp_on_msg = project_p1_1d_sz(
            sd_low,
            mg_side,
            p_low_frac,
            tol=tol,
            rotation_matrix=rot_matrix,
            dim_bool=dim_bool,
            rotate_source=True,  # Fracture grid is in physical coordinates.
            rotate_target=True,  # Mortar side grid is in physical coordinates.
        )

        # (3) side scalars on mortar side grid
        k_side = P_msg @ k_mortar  # (n_msg_cells, 1)
        nv_side = P_msg @ normal_vel_mortar  # (n_msg_cells, 1)

        # (4) jump and integration on mortar side (1D)
        deltap_side = fracp_on_msg - tracep_on_msg  # (n_msg_cells, n_p1_dofs)

        elements = mdnme.utils.get_quadpy_elements(
            mg_side,
            rotate_grid=True,
            rotation_matrix=rot_matrix,
        )

        def integrand(x):
            coors = x[np.newaxis, :, :]  # add new axis, this is needed for 1D grids
            # x is in the interface frame; evaluate P1 on mortar-side cells
            p_jump = mdnme.utils.evaluate_p1(deltap_side, coors)  # (n_msg_cells, n_pts)
            # Broadcasting over quadrature points:
            return (k_side ** (-0.5) * nv_side + k_side**0.5 * p_jump) ** 2

        diff_side = method.integrate(integrand, elements)  # (n_msg_cells,)

        # (5) scatter to global mortar ordering
        out_global += (P_msg.T @ diff_side).ravel()

    return out_global


def _interface_diffusive_error_2d_nonmatching(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
    tol: float = 1e-5,
) -> np.ndarray:
    """Non-matching 2D interface diffusive error:
    || k^{-1/2} λ + k^{1/2} (p_low − tr p_high) ||^2 per mortar cell.
    Requires assign_canonical_rotations() so intf.rot_matrix is set."""
    # --- sanity ---
    if intf.dim != 2:
        raise ValueError("Expected two-dimensional interface grid.")

    # Get canonical frame of rotation
    rot_matrix, dim_bool, _ = mdnme.canonical_frame(intf)

    # --- mortar-side scalars ---
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    k_mortar = (
        float(eff_perm) * np.ones((intf.num_cells, 1))
        if np.isscalar(eff_perm)
        else np.asarray(eff_perm, dtype=float).reshape(-1, 1)
    )
    normal_vel_mortar = _get_normal_velocity(intf, data_intf)  # (n_mortar, 1)

    # --- low-dim pressure (per-cell P1 on its own grid) ---
    p_low_frac = data_low["estimates"]["recon_sd_pressure"]  # (n_frac_cells, 3)

    # --- face-trace of high-dim pressure, in interface frame ---
    # NOTE: p_trace_high[i] corresponds to sd_high face index frac_faces[i]
    frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
    p_trace_high = _get_high_pressure_trace(
        sd_low, sd_high, data_high, frac_faces, rot_matrix, dim_bool
    )  # (n_frac_faces, 3)
    # map: high face id -> local index into frac_faces
    face2pos = {int(f): i for i, f in enumerate(frac_faces)}

    # --- IBG (per-side internal-boundary grids in interface frame) ---
    ibg = InternalBoundaryGrid(intf, sd_high, tol=tol)

    # --- quadrature on 2D mortar sides ---
    method = quadpy.t2.get_good_scheme(4)

    # accumulator in global mortar ordering
    out_global = np.zeros(intf.num_cells)

    # loop sides in the mortar’s canonical order
    for P_msg, mg_side in intf.project_to_side_grids():
        # identify the side enum that owns this mortar side grid
        side_enum = next(k for k, v in intf.side_grids.items() if v is mg_side)

        # IBG side grid and its parent faces (one parent face per IBG triangle)
        ibg_side = ibg.ibg_side_grid(side_enum)
        parent_faces = ibg.parent_face_of_cell(side_enum)  # shape: (n_ibg_cells,)

        # (1) IBG-side tr(p_high): assign face P1 coeffs to each IBG cell via parent map
        if ibg_side.num_cells == 0:
            tr_hi_on_ibg = np.zeros((0, 3))
        else:
            idx = np.fromiter(
                (face2pos[int(f)] for f in parent_faces),
                dtype=int,
                count=parent_faces.size,
            )
            tr_hi_on_ibg = p_trace_high[idx, :]  # (n_ibg_cells, 3)

        # (2) Transfer IBG→mortar-side and frac→mortar-side
        tg_ibg_msg = TransferGrid(
            g_source=ibg_side, g_target=mg_side, rotation_matrix=rot_matrix, tol=tol
        )

        tg_fg_msg = TransferGrid(
            g_source=sd_low, g_target=mg_side, rotation_matrix=rot_matrix, tol=tol
        )

        # Internal boundary side grid to mortar side grid pressure projection
        tracep_on_tg = prolong_to_transfer(tg_ibg_msg, tr_hi_on_ibg)
        tracep_on_msg = scott_zhang_quasi_interpolant(tg_ibg_msg, tracep_on_tg)

        # Fracture grid to mortar side grid pressure projection
        fracp_on_tg = prolong_to_transfer(tg_fg_msg, p_low_frac)
        fracp_on_msg = scott_zhang_quasi_interpolant(tg_fg_msg, fracp_on_tg)

        # (3) side scalars on mortar side grid
        k_side = P_msg @ k_mortar  # (n_msg_cells, 1)
        nv_side = P_msg @ normal_vel_mortar  # (n_msg_cells, 1)

        # (4) jump and integration on mortar side
        deltap_side = fracp_on_msg - tracep_on_msg  # (n_msg_cells, 3)

        elements = mdnme.utils.get_quadpy_elements(
            mg_side,
            rotate_grid=True,
            rotation_matrix=rot_matrix,
        )

        def integrand(x):
            # x is in the interface frame (as provided by elements)
            p_jump = mdnme.utils.evaluate_p1(
                deltap_side, x
            )  # shape broadcast over points
            return (k_side ** (-0.5) * nv_side + k_side**0.5 * p_jump) ** 2

        diff_side = method.integrate(integrand, elements)  # (n_msg_cells,)

        # (5) scatter to global mortar ordering
        out_global += (P_msg.T @ diff_side).ravel()

    return out_global
