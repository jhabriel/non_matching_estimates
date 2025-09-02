from __future__ import annotations

import numpy as np

import mdnme
import porepy as pp


class RotatedGrid:
    """Parent class for rotated grid object."""

    def __init__(self,
                 sd: pp.Grid,
                 rotation_matrix: np.ndarray | None = None,
                 tol=1e-5,  # same as `map_geometry.map_grid()` by default
                 ):
        """
        This class is a wrapper of the outputs of pp.map_geometry.map_grid(sd).

        Alternatively, if `rotation_matrix` (3x3) is supplied, we use it to rotate
        the subdomain grid `sd`.

        Public Attributes:
        -------------------
            sd: Original pp.Grid
            cell_centers: Cell centers coordinates of the rotated grid.
            face_normals: Face normals of the rotated grid.
            face_centers: Face centers coordinates of the rotated grid.
            rotation_matrix: Rotation matrix used to map from the original to the
                rotated domain.
            dim_bool: Array of booleans, where True refers to an active dimension.
            dim: Effective dimension of the rotated grid.
            nodes: Nodes coordinates of the rotated grid.

        """
        if rotation_matrix is not None:
            if rotation_matrix.shape != (3, 3):
                raise ValueError("Expected `rotation_matrix` of shape (3, 3).")

        if rotation_matrix is None:
            # Wrap outputs of the `pp.map_geometry.map_grid(sd)`
            cc, fn, fc, rot_mat, dim_bool, nodes = pp.map_geometry.map_grid(sd)
            self.sd: pp.Grid = sd
            self.cell_centers: np.ndarray = cc
            self.face_normals: np.ndarray = fn
            self.face_centers: np.ndarray = fc
            self.nodes: np.ndarray = nodes
            self.rotation_matrix: np.ndarray = rot_mat
            self.dim_bool: np.ndarray = dim_bool
            self.dim: int = sum(self.dim_bool)
        else:
            self.sd = sd

            # use the provided rotation matrix
            self.rotation_matrix = rotation_matrix

            full_nodes = rotation_matrix @ sd.nodes
            full_cell_centers = rotation_matrix @ sd.cell_centers
            full_face_centers = rotation_matrix @ sd.face_centers
            full_face_normals = rotation_matrix @ sd.face_normals

            spans = np.ptp(full_nodes, axis=1)
            dim_bool = spans > tol
            self.nodes = full_nodes[dim_bool, :]
            self.cell_centers = full_cell_centers[dim_bool, :]
            self.face_centers = full_face_centers[dim_bool, :]
            self.face_normals = full_face_normals[dim_bool, :]
            self.dim_bool = dim_bool
            self.dim = int(dim_bool.sum())

    def __str__(self):
        return "Rotated pseudo-grid object."

    def __repr__(self):
        return (
            "Rotated pseudo-grid object with attributes:\n"
            + "cell_centers\n"
            + "face_normals\n"
            + "face_centers\n"
            + "rotation_matrix\n"
            + "dim\n"
            + "dim_bool\n"
            + "nodes"
        )


def assign_canonical_rotations(mdg: pp.MixedDimensionalGrid) -> None:
    """Transverses the mdg (bottom-up) and assigns canonical rotations all grids.

    This is done by appending the .rot_matrix and .dim_bool to all (side)-grids
    in the mixed-dimensional grid.

    Canonical rotations are defined according to the following rules:

    (i)   Highest-dimensional grids uses .rot_matrix = np.eye(3) and
          .dim_bool = [True, True, True].

    (ii)  Zero-dimensional subdomains and interface grids are assigned
          .rot_matrix = None and .dim_bool = [0, 0, 0]

    (iii) 1d subdomain and 2d subdomain grids use their natural rotation, given by
           mdnme.RotatedGrid(sd).rotation_matrix and mdnme.RotatedGrid(sd).dim_bool

    (iv) 1d and 2d Interface grids inherit the rotation matrices of their
          equidimensional neighboring subdomain (e.g., the one computed in (iii)).

    """

    # Implement rule (i): top-dim
    for sd in mdg.subdomains(dim=mdg.dim_max()):
        sd.rot_matrix = np.eye(3)
        sd.dim_bool = [True, True, True]

    # Implement rule (ii): 0d domains
    for sd in mdg.subdomains(dim=0):
        sd.rot_matrix = None
        sd.dim_bool = [False, False, False]
    for intf in mdg.interfaces(dim=0):
        intf.rot_matrix = None
        intf.dim_bool = [False, False, False]

    # Implement rule (iii): 1d and 2d domains
    for dim in [1, 2]:
        for sd in mdg.subdomains(dim=dim):
            sd_rot = mdnme.RotatedGrid(sd)
            rot_matrix = sd_rot.rotation_matrix
            dim_bool = sd_rot.dim_bool
            sd.rot_matrix = rot_matrix
            sd.dim_bool = dim_bool

    # Implement rule (iv): intfs inherit from their equidimensional sd neighbor
    for dim in [1, 2]:
        for intf in mdg.interfaces(dim=dim):
            sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
            assert intf.dim == sd_low.dim
            intf.rot_matrix = sd_low.rot_matrix
            intf.dim_bool = sd_low.dim_bool