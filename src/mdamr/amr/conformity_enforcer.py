"""
Module containing functionality to enforce interdimensional conformity among md-grids.
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps


def get_coupling_matrix(
        sd_primary: pp.Grid,
        intf: pp.MortarGrid,
        sd_secondary: pp.Grid,
        ) -> np.ndarray:
    """Computes the coupling matrix containing the cell indices of the cells.

    Intended use is to check whether an interdimensional coupling is conforming or not.

    Parameters:
        sd_primary: pp.Grid
            Higher-dimensional grid.
        intf: pp.MortarGrid
            Interface grid.
        sd_secondary: pp.Grid
            Lower-dimensional grid.

    Returns:
        A num_cells-by-3 array, containing the cell indices of the
        highest-dimensional grid (first row), interface grid (second row),
        and lower-dimensional grid (third row).

    """
    # ---> Obtain mapping between mortar and higher-dimensional grid
    sd_primary_faces, intf_cells_high, _ = sps.find(intf.mortar_to_primary_int())

    # Sort indices in ascending order, taking the interface grid as a proxy
    sorted_intf_idx_high = np.argsort(intf_cells_high)
    sorted_sd_primary_faces = sd_primary_faces[sorted_intf_idx_high]
    sorted_intf_cells_high = intf_cells_high[sorted_intf_idx_high]

    # Since the refinement strategy needs to know the cell index, we retrive the cell
    # corresponding to the higher-dimensional face
    sorted_sd_primary_cells = sps.find(
        sd_primary.cell_faces[sorted_sd_primary_faces]
        )[1]

    # Obtain mapping between mortar and lower-dimensional grid
    sd_secondary_cells, intf_cells_low, _ = sps.find(intf.mortar_to_secondary_int())

    # Again, sort indices in ascending order, taking the interface grid as a proxy
    sorted_intf_idx_low = np.argsort(intf_cells_low)
    sorted_sd_secondary_cells = sd_secondary_cells[sorted_intf_idx_low]
    sorted_intf_cells_low = intf_cells_low[sorted_intf_idx_low]

    # Make sure the interface cells obtain from the higher and lower dimensional
    # coupling match exactly
    np.testing.assert_equal(sorted_intf_cells_low, sorted_intf_cells_high)
    sorted_intf_cells = sorted_intf_cells_high

    # Now we can create a num_intf_cells-by-3 matrix containing the full coupling
    coupling_matrix = np.array(
        [sorted_sd_primary_cells, sorted_intf_cells, sorted_sd_secondary_cells]
    )

    # Make sure we have the right shape
    assert coupling_matrix.shape == (3, intf.num_cells)

    return coupling_matrix
