"""Module containing functionality to transform from FEM grid to PorePy griid"""

from __future__ import annotations

import numpy as np

import mdamr
import porepy as pp


def fem_grid_to_sd_grid_1d(
        coordinates: np.ndarray,
        elements: np.ndarray,
        original_grid: pp.Grid,
        ) -> pp.TensorGrid:
    """Converts from a FEM grid to a PorePy grid.

        Parameters:
            coordinates: coordinates of the vertices. Shape is (num_vertices, 2).
            elements: cell-node connectivity map.
                Shape is (num_vertices + 1, num_cells).
            original_grid: Original (coarsest) grid. An instance of pp.Grid.
    """
    g_rot = mdamr.RotatedGrid(original_grid)
    active_dimension = int(np.where(g_rot.dim_bool)[0])
    R = g_rot.rotation_matrix
    local_coordinates = np.zeros([coordinates.shape[0], 3]).T
    local_coordinates[active_dimension] = coordinates.T[0]
    global_coordinates_active_dim = np.dot(np.linalg.inv(R), local_coordinates)
    print("Here")






def fem_grid_to_sd_grid_2d(
        coordinates: np.ndarray,
        elements: np.ndarray,
        ) -> pp.TriangleGrid:
    """Converts from a FEM grid to a PorePy grid with computed geometry.

    Parameters:
        coordinates: np.ndarray
            Coordinates of the vertices (nodes). Shape is (num_elements x dim).
        elements: np.ndarray
            Simplices defined by the nodes numbers. Shape is
            (num_elements x (dim + 1)). We assume that each element is positively
            oriented.

    Returns:
        PorePy triangle grid.

    """
    # We have to transpose the input data
    # TODO: Consider using the same format for FEM grids and PorePy grids
    points = coordinates.T
    triangles = elements.T
    # Create grid
    g = pp.TriangleGrid(p=points, tri=triangles)
    # Compute geometry
    g.compute_geometry()
    return g
