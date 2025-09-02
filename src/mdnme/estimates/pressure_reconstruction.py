"""
This module contains functions that performs pressure reconstructions. The basic idea is
to apply an interpolator of the type G: P_0 -> P_1 to enhance the regularity of the
cell-centered (P0) pressures and obtain P_1, energy-conforming potentials.

Currently, two reconstruction techniques are employed:

    - keilegavlen_p1: This scheme uses the inverse of the reconstructed (RT0) numerical
      fluxes to approximate the pressure gradient at the barycenter of each cell,
      which is later projected to the Lagrangian nodes of the P1 simplex.
    - patchwise_p1: This scheme performs volume-weighted average of the cell-centered
      pressures conforming a patch.

Both methods assume continuity of pressures on Dirichlet boundaries.

"""
from __future__ import annotations

import numpy as np
import porepy as pp
import scipy.sparse as sps

import mdnme


def reconstruct_pressure(mdg: pp.MixedDimensionalGrid, method: str) -> None:
    """Reconstructs the pressure in all subdomains of the mixed-dimensional grid.

    The data dictionary of each node of the grid bucket is updated with the field
    d["estimates"]["recon_sd_pressure"], a NumPy nd-array containing the coefficients
    of the reconstructed pressure.

    Parameters:
        mdg : pp.MixedDimensionalGrid
            Mixed-dimensional grid.
        method : str
            Pressure reconstruction method. Either 'keilegavlen_p1' or 'patchwise_p1'.

    """
    # Loop through all subdomains
    for sd, sd_data in mdg.subdomains(return_data=True):
        # Create key if it does not exist
        if sd_data["estimates"].get("recon_sd_pressure") is None:
            sd_data["estimates"]["recon_sd_pressure"] = {}

        # Handle the case of zero-dimensional subdomains
        if sd.dim == 0:
            sd_data["estimates"]["recon_sd_pressure"] = sd_data["estimates"][
                "fv_sd_pressure"
            ]
            continue

        # Retrieve boundary grid and data associated to the subdomain
        bg = mdg.subdomain_to_boundary_grid(sd)
        assert isinstance(bg, pp.BoundaryGrid)  # to please mypy
        bg_data = mdg.boundary_grid_data(bg)

        # Obtain reconstructed pressure values with their Lagrangian coordinates
        if method == "keilegavlen_p1":
            point_val, point_coo = keilegavlen_p1(sd, sd_data, bg_data)
        elif method == "patchwise_p1":
            point_val, point_coo = patchwise_p1(sd, sd_data, bg_data)
        else:
            raise ValueError("Pressure reconstruction method not implemented.")

        # Obtain pressure coefficients
        recons_p = mdnme.utils.interpolate_p1(point_val, point_coo)

        # # TEST: Pressure reconstruction
        # self._test_pressure_reconstruction(sd, recons_p, point_val, point_coo)
        # # END TEST

        # Save in the data dictionary.
        sd_data["estimates"]["recon_sd_pressure"] = recons_p


def patchwise_p1(
    sd: pp.Grid, sd_data: dict, bg_data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Pressure reconstruction using average of P0(K) potentials over patches [1].

    Reference:
        Cochez-Dhondt, S., Nicaise, S., & Repin, S. (2009). A posteriori error estimates
        for finite volume approximations. Mathematical Modelling of Natural Phenomena,
        4(1), 106-122.

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        sd_data: dict
            Subdomain data dictionary.
        bg_data: dict
            Boundary grid data dictionary.

    Returns:
        2-tuple of numpy arrays containing the values and Lagrangian coordinates of
        the reconstructed pressure for all elements of the subdomain grid.

    """
    # Retrieve finite volume cell-centered pressures
    p_cc = sd_data["estimates"]["fv_sd_pressure"]
    assert p_cc.size == sd.num_cells

    # Rotated grid
    sd_rot = mdnme.RotatedGrid(sd)

    # Retrieving topological data
    nc = sd.num_cells
    nf = sd.num_faces

    # Perform reconstruction
    cell_nodes = sd.cell_nodes()
    cell_nodes_volume = cell_nodes * sps.dia_matrix((sd.cell_volumes, 0), (nc, nc))
    cell_nodes_pressure = cell_nodes * sps.dia_matrix((p_cc, 0), (nc, nc))

    numerator = cell_nodes_volume.multiply(cell_nodes_pressure)
    numerator = np.array(numerator.sum(axis=1)).flatten()
    denominator = np.array(cell_nodes_volume.sum(axis=1)).flatten()
    nodal_pressures = numerator / denominator

    # Treatment of boundary conditions
    bc = sd_data[pp.PARAMETERS]["flow"]["bc"]

    bc_dir_values = np.zeros(sd.num_faces)
    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, sd.tags["domain_boundary_faces"]
    )
    bg_dict = bg_data[pp.ITERATE_SOLUTIONS]
    bc_pressure = bg_dict["pressure"][0]
    bg_dir_filter = bg_dict["bc_values_darcy_flux_filter_dir"][0] == 1
    bc_dir_values[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = sd.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_dir_values[external_dirichlet_boundary]
    node_val_dir = sd.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

    # Export Lagrangian nodes and coordinates
    # cell_nodes_map, _, _ = sps.find(sd.cell_nodes())
    # nodes_cell = cell_nodes_map.reshape(np.array([sd.num_cells, sd.dim + 1]))
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(sd.num_cells, sd.dim + 1)
    point_val = nodal_pressures[nodes_of_cell]
    point_coo = sd_rot.nodes[:, nodes_of_cell]

    return point_val, point_coo


def keilegavlen_p1(
    sd: pp.Grid, sd_data: dict, bg_data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Pressure reconstruction using the inverse of the numerical fluxes.

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        sd_data: dict
            Subdomain data dictionary.
        bg_data: dict
            Boundary grid data dictionary.

    Returns:
        2-tuple of numpy arrays containing the values and Lagrangian coordinates of
        the reconstructed pressure for all elements of the subdomain grid.

    """
    # Retrieve finite volume cell-centered pressures
    p_cc = sd_data["estimates"]["fv_sd_pressure"]
    assert p_cc.size == sd.num_cells

    # Rotated grid
    sd_rot = mdnme.RotatedGrid(sd)

    # Retrieve topological data
    nc = sd.num_cells
    nf = sd.num_faces
    nn = sd.num_nodes

    # Perform reconstruction
    cell_nodes = sd.cell_nodes()
    cell_node_volumes = cell_nodes * sps.dia_matrix(
        arg1=(sd.cell_volumes, 0), shape=(nc, nc)
    )
    sum_cell_nodes = cell_node_volumes * np.ones(nc)
    cell_nodes_scaled = (
        sps.dia_matrix(arg1=(1.0 / sum_cell_nodes, 0), shape=(nn, nn))
        * cell_node_volumes
    )

    # Retrieve reconstructed velocities
    coeff = sd_data["estimates"]["recon_sd_flux"]
    if sd.dim == 3:
        proj_flux = np.array(
            [
                coeff[:, 0] * sd_rot.cell_centers[0] + coeff[:, 1],
                coeff[:, 0] * sd_rot.cell_centers[1] + coeff[:, 2],
                coeff[:, 0] * sd_rot.cell_centers[2] + coeff[:, 3],
            ]
        )
    elif sd.dim == 2:
        proj_flux = np.array(
            [
                coeff[:, 0] * sd_rot.cell_centers[0] + coeff[:, 1],
                coeff[:, 0] * sd_rot.cell_centers[1] + coeff[:, 2],
            ]
        )
    else:
        proj_flux = np.array(
            [
                coeff[:, 0] * sd_rot.cell_centers[0] + coeff[:, 1],
            ]
        )

    # Obtain local gradients
    loc_grad = np.zeros((sd.dim, nc))
    perm = sd_data[pp.PARAMETERS]["flow"]["second_order_tensor"].values
    for ci in range(nc):
        loc_grad[: sd.dim, ci] = -np.linalg.inv(perm[: sd.dim, : sd.dim, ci]).dot(
            proj_flux[:, ci]
        )

    # Obtaining nodal pressures
    cell_nodes_map = sps.find(sd.cell_nodes().T)[1]
    cell_node_matrix = cell_nodes_map.reshape(np.array([sd.num_cells, sd.dim + 1]))
    nodal_pressures = np.zeros(nn)

    for col in range(sd.dim + 1):
        nodes = cell_node_matrix[:, col]
        dist = sd_rot.nodes[: sd.dim, nodes] - sd_rot.cell_centers[: sd.dim]
        scaling = cell_nodes_scaled[nodes, np.arange(nc)]
        contribution = (
            np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
        ).ravel()
        nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

    # Treatment of boundary conditions
    bc = sd_data[pp.PARAMETERS]["flow"]["bc"]

    bc_dir_values = np.zeros(sd.num_faces)
    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, sd.tags["domain_boundary_faces"]
    )
    bg_dict = bg_data[pp.ITERATE_SOLUTIONS]
    bc_pressure = bg_dict["pressure"][0]
    bg_dir_filter = bg_dict["bc_values_darcy_flux_filter_dir"][0] == 1
    bc_dir_values[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = sd.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_dir_values[external_dirichlet_boundary]
    node_val_dir = sd.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

    # Export Lagrangian nodes and coordinates
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(sd.num_cells, sd.dim + 1)
    point_val = nodal_pressures[nodes_of_cell]
    point_coo = sd_rot.nodes[:, nodes_of_cell]

    return point_val, point_coo
