"""
Converter between PorePy and Coordinate-Elements grid formats.
"""

# Import modules
import numpy as np

import mdnme
import porepy as pp
import scipy.sparse as sps

from mdnme.amr.refinement_utils import enforce_positive_orientation


def porepy_grid_to_fem_grid(g: pp.GridLike) -> tuple[np.ndarray, np.ndarray]:
    """Transforms a PorePy Grid into a FEM grid.

    Parameters:
        ....

    Returns:
        ...

    """
    if g.dim == 3:
        ...
        # assert isinstance(g, pp.Grid) and isinstance(g, pp.TetrahedralGrid)
        # coo, ele = subdomain_grid_3d_to_fem_grid(g)
    elif g.dim == 2:
        if isinstance(g, pp.Grid) and isinstance(g, pp.TriangleGrid):
            coo, ele = subdomain_grid_2d_to_fem_grid(g)
        elif isinstance(g, pp.MortarGrid):
            ...
            # coo, ele = mortar_grid_2d_to_fem_grid()
        elif isinstance(g, pp.BoundaryGrid):
            ...
            # coo, ele = boundary_grid_2d_to_fem_grid()
        else:
            ValueError("PorePy grid type not supported.")
    elif g.dim == 1:
        if isinstance(g, pp.Grid):
            coo, ele = subdomain_grid_1d_to_fem_grid(g)
        elif isinstance(g, pp.MortarGrid):
            coo, ele = mortar_grid_1d_to_fem_grid(g)
        elif isinstance(g, pp.BoundaryGrid):
            ...
            # coo, ele = boundary_grid_1d_to_fem_grid()
        else:
            ValueError("PorePy grid type not supported.")
    elif g.dim == 0:
        pass  # nothing to do here, the grid is 0d
    else:
        ValueError("Expected grid of dimension 0, 1, 2, or 3.")

    return coo, ele


def subdomain_grid_3d_to_fem_grid():
    ...


def subdomain_grid_2d_to_fem_grid(g: pp.TriangleGrid) -> tuple[np.ndarray, np.ndarray]:
    """Transforms a 2d subdomain grid into a FEM grid.

    Parameters:
        g: pp.TriangleGrid
            Delauney grrid.

    Returns:
        ...
        ...

    """
    # Retrieve node coordinates
    coordinates = g.nodes.transpose()[:, :g.dim]

    # Retrieve node-triangles mapping
    elements = g.cell_node_matrix()

    # In some situations, PorePy grids (strangely) have elements whose nodes are not
    # positively oriented. This has devastating effects in the adaptive mesh
    # refinement process (breaking at worst and loose of shape regularity at best).
    # Thus, we add an extra step where we ensure all nodes to be strictly positively
    # oriented. This is easily achieved in 2d be swapping any two-vertices.
    positively_oriented_elements = enforce_positive_orientation(coordinates, elements)

    return coordinates, positively_oriented_elements


def subdomain_grid_1d_to_fem_grid(g: pp.Grid) -> tuple[np.ndarray, np.ndarray]:

    g_rot = mdnme.RotatedGrid(g)
    coordinates = g_rot.nodes.transpose()[:, :1]
    elements = np.reshape(sps.find(g.cell_nodes())[0], [g.num_cells, 2])

    # TODO: Do we need to check for positively oriented direction?

    return coordinates, elements


def mortar_grid_2d_to_fem_grid():
    ...


def mortar_grid_1d_to_fem_grid(intf: pp.MortarGrid) -> tuple[np.ndarray, np.ndarray]:

    # Things are not so direct with mortar grids. We need to work with the side grids
    side0, side1 = intf.project_to_side_grids()

    cells_side0 = sps.find(side0[0])[1]
    g_side0 = side0[1]

    cells_side1 = sps.find(side1[0])[1]
    g_side1 = side1[1]

    # Now we need to rotate the grids
    g_side0_rot = mdnme.RotatedGrid(g_side0)
    g_side1_rot = mdnme.RotatedGrid(g_side1)

    # Obtain the coordinates
    coordinates_side0 = g_side0_rot.nodes.transpose()[:, :1]
    coordinates_side1 = g_side1_rot.nodes.transpose()[:, :1]
    coordinates = np.vstack((coordinates_side0, coordinates_side1))

    # Obtain element-coordinates mapping
    elements_side0 = np.reshape(
        sps.find(g_side0.cell_nodes())[0], [g_side0.num_cells,2]
    )

    elements_side1 = np.reshape(
        sps.find(g_side1.cell_nodes())[0], [g_side1.num_cells,2]
    )

    # TODO: This is currently a hack. Not sure if this going to work generally
    elements_side0 += cells_side0[0]
    elements_side1 += cells_side1[0] + 1
    elements = np.vstack((elements_side0, elements_side1))

    # TODO: Check for positive orientation?

    return coordinates, elements

def boundary_grid_2d_to_fem_grid():
    ...


def boundary_grid_1d_to_fem_grid():
    ...


def porepy_grid_to_fem_mesh(g: pp.GridLike) -> tuple[np.ndarray, np.ndarray]:
    """Converts a PorePy grid to a finite element mesh.

    Parameters:
        g: pp.Grid
            PorePy grid with computed geometry. Can be either an interface or a
            subdomain grid.

    Returns:
        coordinates: np.ndarray
            Coordinates of the vertices defining each simplex. Shape is
            (num_elements, g.dim).
        elements: np.ndarray
            Vertices defining the simplex. Shape is (num_elements, g.dim + 1).

    """
    coordinates = g.nodes.transpose()[:, :g.dim]
    elements = g.cell_node_matrix()
    return  coordinates, elements
