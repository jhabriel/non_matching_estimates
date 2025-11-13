"""
This module contains functionality to extend normal fluxes on edges (typically coming
from finite-volume discretizations) onto the interior of the elements using
lowest-order Raviart-Thomas (RT0) basis functions.

"""
from __future__ import annotations

import numpy as np
import porepy as pp
import scipy.sparse as sps

import mdnme


def extend_fv_fluxes(mdg: pp.MixedDimensionalGrid) -> None:
    """Extend normal fluxes using RT0 basis functions.

    Parameters:
        mdg: pp.MixedDimensionalGrid
            Mixed-dimensional grid for the problem.

    Note:
        The data dictionary of each node of the grid bucket will be updated with the
        field d["estimates"]["recon_sd_flux"], a nd-array of shape
        (sd.num_cells x (sd.dim+1)) containing the coefficients of the reconstructed
        flux for each element. Each column corresponds to the coefficient a, b, c,
        and so on.

        The coefficients satisfy the following velocity fields depending on the
        dimensionality of the problem:

            q = ax + b                          (for 1d),
            q = (ax + b, ay + c)^T              (for 2d),
            q = (ax + b, ay + c, az + d)^T      (for 3d).

        The reconstructed velocity field inside an element K is given by:

            q = sum_{j=1}^{g.dim+1} q_j psi_j,

        where psi_j are the global basis functions defined on each face,
        and q_j are the normal fluxes.

        The global basis takes the form

            psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i)^T                     (for 1d),
            psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i)^T            (for 2d),
            psi_j = (s(normal_j)/(g.dim|K|)) (x - x_i, y - y_i, z - z_i)^T   (for 3d),

        where s(normal_j) is the sign of the normal vector,|K| is the Lebesgue
        measure of the element K, and (x_i, y_i, z_i) are the coordinates of the
        opposite side nodes to the face j. The function s(normal_j) = 1 if the
        signs of the local and global normals are the same, and -1 otherwise.

    """
    # Loop through all the nodes of the grid bucket
    for sd, d in mdg.subdomains(return_data=True):
        # Create key if it does not exist
        if d["estimates"].get("recon_sd_flux") is None:
            d["estimates"]["recon_sd_flux"] = {}

        # Handle the case of zero-dimensional subdomains
        if sd.dim == 0:
            d["estimates"]["recon_sd_flux"] = None
            continue

        # Rotate the grid (no effect on highest-dimensional subdomains)
        sd_rot = mdnme.rotate_grid(sd)

        # Cell-basis arrays
        cell_faces_map = sps.find(sd.cell_faces.T)[1]
        faces_cell = cell_faces_map.reshape(sd.num_cells, sd.dim + 1)
        opp_nodes_cell = get_opposite_side_nodes(sd)
        opp_nodes_coor_cell = sd_rot.nodes[:, opp_nodes_cell]
        sign_normals_cell = get_sign_normals(sd)
        vol_cell = sd.cell_volumes

        # Retrieve finite volume fluxes
        flux = d["estimates"]["fv_sd_flux"]

        # TEST -> Local mass conservation
        # Check if mass conservation is satisfied on a cell basis, in order to do
        # this, we check on a local basis, if the divergence of the flux equals
        # the sum of internal and external source terms
        # full_flux_local_div = (sign_normals_cell * flux[faces_cell]).sum(axis=1)
        # external_src = d[pp.PARAMETERS][self.kw]["source"]
        # np.testing.assert_allclose(
        #     full_flux_local_div,
        #     external_src + mortar_jump,
        #     rtol=1e-6,
        #     atol=1e-3,
        #     err_msg="Error estimates only valid for local mass-conservative methods.",
        # )
        # END OF TEST

        # Perform actual reconstruction and obtain coefficients
        coeffs = np.empty([sd.num_cells, sd.dim + 1])
        alpha = 1 / (sd.dim * vol_cell)
        coeffs[:, 0] = alpha * np.sum(sign_normals_cell * flux[faces_cell], axis=1)
        for dim in range(sd.dim):
            coeffs[:, dim + 1] = -alpha * np.sum(
                (sign_normals_cell * flux[faces_cell] * opp_nodes_coor_cell[dim]),
                axis=1,
            )

        # Store coefficients in the data dictionary
        d["estimates"]["recon_sd_flux"] = coeffs


def get_opposite_side_nodes(sd: pp.Grid) -> np.ndarray:
    """Computes opposite side nodes for each face of each cell in the grid.

    Parameters:
        sd: pp.Grid
            Subdomain grid.

    Returns:
        Opposite nodes with rows representing the cell number and columns
        representing the opposite side node index of the face. The size of the array
        is (sd.num_cells x (sd.dim + 1)).

    """
    dim = sd.dim
    nc = sd.num_cells
    nf = sd.num_faces

    faces_of_cell = sps.find(sd.cell_faces.T)[1].reshape(nc, dim + 1)
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(nc, dim + 1)
    nodes_of_face = sps.find(sd.face_nodes.T)[1].reshape(nf, dim)

    opposite_nodes = np.empty_like(faces_of_cell)
    for cell in range(sd.num_cells):
        opposite_nodes[cell] = [
            np.setdiff1d(nodes_of_cell[cell], nodes_of_face[face])[0]
            for face in faces_of_cell[cell]
        ]

    return opposite_nodes


def get_sign_normals(sd: pp.Grid) -> np.ndarray:
    """Computes sign of the face normals for each cell of the grid.

    Note:
        We have to take care of the sign of the basis functions. The idea is to create
        an array of signs "sign_normals" that will be multiplying each edge basis
        function for the RT0 extension of fluxes.

        To determine this array, we need the following:
            (1) Compute the local outer normal `lon` vector for each cell.
            (2) For every face of each cell, compare if lon == global normal vector.
                If they're not, then we need to flip the sign of lon for that face

    Parameters:
        sd: pp.Grid
            Subdomain grid.

    Returns:
        Sign of the face normal. 1 if the signs of the local and global normals are
        the same, -1 otherwise. The size of the np.ndarray is `sd.num_faces`.

    """
    # Faces associated to each cell
    faces_cell = sps.find(sd.cell_faces.T)[1].reshape(sd.num_cells, sd.dim + 1)

    # Face centers coordinates for each face associated to each cell
    sd_rot = mdnme.rotate_grid(sd)
    face_center_cells = sd_rot.face_centers[:, faces_cell]

    # Global normals of the faces per cell
    global_normal_faces_cell = sd_rot.face_normals[:, faces_cell]

    # Computing the local outer normals of the faces per cell. To do this, we first
    # assume that n_loc = n_glb, and then we fix the sign. To fix the sign,we compare
    # the length of two vectors, the first vector v1 = face_center - cell_center,
    # and the second vector v2 is a prolongation of v1 in the direction of the
    # normal. If ||v2||<||v1||, then the  normal of the face in question is pointing
    # inwards, and we needed to flip the sign.
    local_normal_faces_cell = global_normal_faces_cell.copy()
    v1 = face_center_cells - sd_rot.cell_centers[:, :, np.newaxis]
    v2 = v1 + local_normal_faces_cell * 0.001
    # Checking if ||v2|| < ||v1|| or not
    length_v1 = np.linalg.norm(v1, axis=0)
    length_v2 = np.linalg.norm(v2, axis=0)
    swap_sign = 1 - 2 * (length_v2 < length_v1)
    # Swapping the sign of the local normal vectors
    local_normal_faces_cell *= swap_sign

    # Now that we have the local outer normals. We can check if the local
    # and global normals are pointing in the same direction. To do this
    # we compute length_sum_n = || n_glb + n_loc||. If they're the same, then
    # length_sum_n > 0. Otherwise, they're opposite and length_sum_n \approx 0.
    sum_n = local_normal_faces_cell + global_normal_faces_cell
    length_sum_n = np.linalg.norm(sum_n, axis=0)
    sign_normals = 1 - 2 * (length_sum_n < 1e-8)

    return sign_normals
