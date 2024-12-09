from __future__ import annotations

import numpy as np
import porepy as pp


class RotatedGrid:
    """Parent class for rotated grid object."""

    def __init__(self, sd: pp.Grid):
        """
        This class is a wrapper of the outputs of pp.map_geometry.map_grid(sd).

        Public Attributes:
        -------------------
            sd: Original pp.Grid
            cell_centers: Cell centers coordinates of the rotated grid.
            face_normals: Face normals of the rotated grid.
            face_centers: Face centers coordinates of the rotated grid.
            rotation_matrix: Rotation matrix used to map from the original to the rotated
                domain.
            dim_bool: Array of booleans, where True refers to an active dimension.
            dim: Effective dimension of the rotated grid.
            nodes: Nodes coordinates of the rotated grid.

        """

        self.sd: pp.Grid = sd
        (
            cell_centers,
            face_normals,
            face_centers,
            rotation_matrix,
            dim_bool,
            nodes,
        ) = pp.map_geometry.map_grid(self.sd)
        self.cell_centers: np.ndarray = cell_centers
        self.face_normals: np.ndarray = face_normals
        self.face_centers: np.ndarray = face_centers
        self.nodes: np.ndarray = nodes
        self.rotation_matrix: np.ndarray = rotation_matrix
        self.dim_bool: np.ndarray = dim_bool
        self.dim: int = sum(self.dim_bool)

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
