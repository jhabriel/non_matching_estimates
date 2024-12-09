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

import mdamr as amr


def compute_diffusive_error(mdg: pp.MixedDimensionalGrid) -> None:
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
            intf, data_intf, sd_high, data_high, sd_low, data_low
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
    elements = amr.utils.get_quadpy_elements(sd)
    if sd.dim == 1:
        method = quadpy.c1.newton_cotes_closed(4)
    elif sd.dim == 2:
        method = quadpy.t2.get_good_scheme(4)
    else:
        method = quadpy.t3.get_good_scheme(4)

    # Obtain coefficients
    p = amr.utils.poly2col(recon_p)
    u = amr.utils.poly2col(recon_u)

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
        diffusive_error = _interface_diffusive_error_1d(
            intf,
            data_intf,
            sd_high,
            data_high,
            sd_low,
            data_low,
        )
    else:
        diffusive_error = _interface_diffusive_error_2d(
            intf,
            data_intf,
            sd_high,
            data_high,
            sd_low,
            data_low,
        )

    return diffusive_error


# Private functions
def _get_high_pressure_trace(
    sd_low: pp.Grid, sd_high: pp.Grid, data_sd_high: dict, frac_faces: np.ndarray
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

    def get_intf_lagrangian_coo(grid: Union[pp.Grid, amr.RotatedGrid]) -> np.ndarray:
        """Gets coordinates of the Lagrangian nodes of the internal higher-dim boundary.

        Parameters:
            grid: pp.Grid or amr.RotatedGrid:
                Higher-dimensional grid.

        Returns:
            Coordinates of the Lagrangian nodes.

        """
        # Get nodes of the fracture faces
        nodes_of_frac_faces = sps.find((sd_high.face_nodes.T)[frac_faces])[1].reshape(
            frac_faces.size, sd_high.dim
        )

        # Obtain the coordinates of the nodes of the fracture faces
        lagran_coo = grid.nodes[:, nodes_of_frac_faces]

        return lagran_coo

    # Rotate both grids, and obtain rotation matrix and effective dimension
    gh_rot = amr.RotatedGrid(sd_high)
    gl_rot = amr.RotatedGrid(sd_low)
    rotation_matrix = gl_rot.rotation_matrix
    dim_bool = gl_rot.dim_bool

    # Obtain the cells corresponding to the frac_faces
    cells_of_frac_faces = sps.find(sd_high.cell_faces[frac_faces])[1]

    # Retrieve the coefficients of the polynomials corresponding to those cells
    if "recon_sd_pressure" in data_sd_high["estimates"]:
        p_high = data_sd_high["estimates"]["recon_sd_pressure"]
    else:
        raise ValueError("Pressure must be reconstructed first")
    p_high = p_high[cells_of_frac_faces]

    # NOTE: Use the rotated coordinates to perform the evaluation of the pressure,
    # but use the original coordinates to rotate the edge using the rotation matrix of
    # the lower-dimensional grid as reference.

    # Evaluate the polynomials at the relevant Lagrangian nodes
    point_coo_rot = get_intf_lagrangian_coo(gh_rot)
    point_val = amr.utils.evaluate_p1(p_high, point_coo_rot)

    # Rotate the coordinates of the Lagrangian nodes w.r.t. the lower-dimensional grid
    point_coo = get_intf_lagrangian_coo(sd_high)
    point_edge_coo_rot = np.empty_like(point_coo)
    for element in range(frac_faces.size):
        point_edge_coo_rot[:, element] = np.dot(rotation_matrix, point_coo[:, element])
    point_edge_coo_rot = point_edge_coo_rot[dim_bool]

    # Construct a polynomial (of reduced dimensionality) using the rotated coo
    trace_pressure = amr.utils.interpolate_p1(point_val, point_edge_coo_rot)

    # Test if the values of the original polynomial match the new one
    point_val_rot = amr.utils.evaluate_p1(trace_pressure, point_edge_coo_rot)
    np.testing.assert_almost_equal(point_val, point_val_rot, decimal=12)

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
    sd_rot = amr.RotatedGrid(sd_high)

    # Obtain the trace of the pressure of the 1D grid
    cells_of_frac_faces = sps.find(sd_high.cell_faces[frac_faces])[1]
    p_1d = data_high["estimates"]["recon_sd_pressure"]
    p_1d = p_1d[cells_of_frac_faces]
    coo_frac_faces = sd_rot.face_centers[:, frac_faces].T
    coo_frac_faces = coo_frac_faces[np.newaxis, :, :]
    trace_p = amr.utils.evaluate_p1(p_1d, coo_frac_faces).flatten()

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
        elements = amr.utils.get_quadpy_elements(sidegrid)

        # Project relevant quantities to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k

        # Declare integrand
        def integrand(x):
            coors = x[np.newaxis, :, :]  # add new axis, this is needed for 1D grids
            p_jump = amr.utils.evaluate_p1(deltap_side, coors)
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
        elements = amr.utils.get_quadpy_elements(sidegrid)

        # Project relevant quantities to the side grid
        deltap_side = projector * deltap
        normalvel_side = projector * normal_vel
        k_side = projector * k

        # Declare integrand
        def integrand(x):
            p_jump = amr.utils.evaluate_p1(deltap_side, x)
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
