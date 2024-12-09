"""
Module for refinement of PorePy grids.
"""

import numpy as np

import mdnme
import porepy as pp
import scipy.sparse as sps

from mdnme.amr.refinement_utils import enforce_positive_orientation

from porepy.grids.mortar_grid import MortarSides


def refine_sd_2d(
    sd: pp.Grid,
    marked_elements: np.ndarray = None,
) -> pp.Grid:
    """Refines a 2d subdomain grid.

    TODO: Consider the case where the grid can be embedded in 3d. Try to follow the
          same philosophy as in the treatment of 1d embedded lines

    Parameters:
        sd: pp.TriangleGrid
            Delaunay triangle grid. An instance of the `pp.TriangleGrid` class.
        marked_elements: np.ndarray
            Array containing marked elements to be refined. Shape is (num_cells, ).

    Returns:
        Refined 2d grid. This grid can be replaced into the original mdg.

    """
    # Check whether the grid is a TriangleGrid
    assert isinstance(sd, pp.TriangleGrid)

    # Check whether grid is 2d
    assert sd.dim == 2
    nc = sd.num_cells

    # Check whether the z-dimension is 0
    if np.mean(np.abs(sd.cell_centers[2])) > 1e-14:
        msg = "A 2d subdomain grid cannot be a fracture grid."
        raise NotImplementedError(msg)

    # If `marked_elements` not given, perform uniform red refinement
    if marked_elements is None:
        marked_elements = np.ones(nc, dtype=bool)

    # Retrieve node coordinates
    coordinates = sd.nodes.transpose()[:, :sd.dim]

    # Retrieve node-triangles mapping
    elements = sd.cell_node_matrix()

    # In some situations, PorePy grids (strangely) have elements whose nodes are not
    # positively oriented. This has devastating effects in the adaptive mesh refinement
    # process (breaking at worst and loose of shape regularity at best). Thus, we add an
    # extra step where we ensure all nodes to be strictly positively oriented. This
    # is easily achieved in 2d be swapping any two-vertices.
    positively_oriented_elements = enforce_positive_orientation(coordinates, elements)

    # Perform refinement
    # TODO: Eventually, we can handle boundary conditions in a more effective way
    new_coordinates, new_elements, *_ = mdamr.refine_rgb(
        coordinates,
        positively_oriented_elements,
        marked_elements
    )

    # Now, we need to produce back a valid TriangleGrid
    points = new_coordinates.T
    triangles = new_elements.T

    # Create grid and compute geometry
    refined_sd = pp.TriangleGrid(p=points, tri=triangles)
    refined_sd.compute_geometry()

    return refined_sd


def refine_intf_1d(
    intf: pp.MortarGrid,
    marked_elements: np.ndarray = None
) -> dict[MortarSides, pp.Grid]:
    """Refines a 1d mortar grid.

    Parameters:
        intf: pp.MortarGrid
            One-dimensional interface grid.
        marked_elements: np.ndarray
            Array of marked interface elements. Shape is (num_intf_cells, ).

    Returns:
        Dictionary containing the refined side grids. Key is the mortar side (an
        integer from the MortarSides class) and value is the sidegrid (a pp.Grid).

    """

    # Check whether grid is 1d
    assert intf.dim == 1
    nc = intf.num_cells

    if intf.num_sides() != 2:
        msg = 'Mortar grid must have two sides. One-sided mortar is yet not implemented'
        raise NotImplementedError(msg)

    if marked_elements is None:
        marked_elements = np.ones(nc, dtype=bool)

    side_left, side_right = intf.project_to_side_grids()

    proj_left, sd_left = side_left
    marked_elements_left = proj_left @ marked_elements

    proj_right, sd_right = side_right
    marked_elements_right = proj_right @ marked_elements

    # Now we can perform the refinement
    new_sd_left = refine_sd_1d(sd_left, marked_elements_left)
    new_sd_right = refine_sd_1d(sd_right, marked_elements_right)

    return {intf.sides[0]: new_sd_left, intf.sides[1]: new_sd_right}


def refine_sd_1d(
    sd: pp.Grid,
    marked_elements: np.ndarray = None
) -> pp.Grid:
    """Refine a 1d subdomain grid.

    Parameters:
        sd: pp.Grid
            One-dimensional PorePy grid.
        marked_elements: np.ndarray
            Array containing marked elements to be refined.

    Returns:
        Refined 1d grid. This grid can be replaced into the original mdg.

    """
    # Check whether grid is 1d
    assert sd.dim == 1
    nc = sd.num_cells

    # If no marked elements are given, refine all elements
    if marked_elements is None:
        marked_elements = np.ones(nc, dtype=bool)

    # Since the refinement is uniform, we can refine independently in each direction
    elements = np.reshape(sps.find(sd.cell_nodes())[0], [nc, 2])

    # Refinement in x-direction
    coo_x = sd.nodes.transpose()[:, 0:1]
    new_coo_x, _ = mdamr.refine_red_1d(coo_x, elements, marked_elements)

    # Refinement in y-direction
    coo_y = sd.nodes.transpose()[:, 1:2]
    new_coo_y, _ = mdamr.refine_red_1d(coo_y, elements, marked_elements)

    # Refinement in z-direction
    coo_z = sd.nodes.transpose()[:, 2:3]
    new_coo_z, _ = mdamr.refine_red_1d(coo_z, elements, marked_elements)

    # New coordinates
    new_coo = np.array([new_coo_x.T[0], new_coo_y.T[0], new_coo_z.T[0]])

    # Now we are in position to create a new PorePy grid
    # This employs the implementation of `create_embedded_line_grid()`
    tol = 1e-6
    loc_center = np.mean(new_coo, axis=1).reshape((-1, 1))
    (
        sorted_coord,
        rot,
        active_dimension,
        sort_ind,
    ) = pp.map_geometry.project_points_to_line(new_coo, tol)
    g = pp.TensorGrid(sorted_coord)

    # Project back to active dimension
    nodes = np.zeros(g.nodes.shape)
    nodes[active_dimension] = g.nodes[0]
    g.nodes = nodes

    # Project back again to 3d coordinates
    irot = rot.transpose()
    g.nodes = irot.dot(g.nodes)
    g.nodes += loc_center

    # Finally, make sure to compute the geometry
    g.compute_geometry()

    return g
