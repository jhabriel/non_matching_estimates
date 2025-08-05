from __future__ import annotations

import numpy as np
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
