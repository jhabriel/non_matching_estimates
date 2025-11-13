from __future__ import annotations

import numpy as np
import porepy as pp
import scipy.sparse as sps

import mdnme as amr


def get_quadpy_elements(
        sd: pp.Grid,
        rotate_grid=True,
        rotation_matrix : np.ndarray | None = None,
) -> np.ndarray:
    """
    Assembles the elements of a given grid in quadpy format: https://pypi.org/project/quadpy/.

    Parameters
    ----------
        sd (pp.Grid): PorePy grid.
        sd_rot (mde.RotatedGrid): Rotated pseudo-grid.

    Returns
    --------
    quadpy_elements (np.ndarray): Elements in QuadPy format.

    Example
    --------
    >>> # shape (3, 5, 2), i.e., (corners, num_triangles, xy_coords)
    >>> triangles = np.stack([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.2, 0.6], [1.3, 0.7], [1.4, 0.8]],
            [[26.0, 31.0], [24.0, 27.0], [33.0, 28]],
            [[0.1, 0.3], [0.4, 0.4], [0.7, 0.1]],
            [[8.6, 6.0], [9.4, 5.6], [7.5, 7.4]]
            ], axis=-2)
    """

    # Renaming variables
    nc = sd.num_cells

    # Getting node coordinates for each cell
    # cell_nodes_map, _, _ = sps.find(sd.cell_nodes())
    # nodes_cell = cell_nodes_map.reshape(np.array([nc, sd.dim + 1]))
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(nc, sd.dim + 1)

    if not rotate_grid:
        nodes_coor_cell = sd.nodes[:, nodes_of_cell]
    else:
        if rotation_matrix is not None:
            rotated_grid = amr.RotatedGrid(sd)
        else:
            rotated_grid = amr.RotatedGrid(sd, rotation_matrix)
        nodes_coor_cell = rotated_grid.nodes[:, nodes_of_cell]

    # Stacking node coordinates
    cnc_stckd = np.empty([nc, (sd.dim + 1) * sd.dim])
    col = 0
    for vertex in range(sd.dim + 1):
        for dim in range(sd.dim):
            cnc_stckd[:, col] = nodes_coor_cell[dim][:, vertex]
            col += 1
    element_coord = np.reshape(cnc_stckd, np.array([nc, sd.dim + 1, sd.dim]))

    # Reshaping to please quadpy format i.e., (corners, num_elements, coords)
    elements = np.stack(element_coord, axis=-2)  # type:ignore

    # For some reason, quadpy needs a different formatting for line segments
    if sd.dim == 1:
        elements = elements.reshape(sd.dim + 1, sd.num_cells)

    return elements
