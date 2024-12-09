"""Red-Green-Blue refinement algorithm in 2D."""


import numpy as np
from mdamr.amr.refinement_utils import hash_to_map, get_edge_data_2d


def refine_rgb(
        coordinates: np.ndarray,
        elements: np.ndarray,
        marked_elements: np.ndarray | None = None,
        dirichlet_bc: np.ndarray | None = None,
        neumann_bc: np.ndarray | None = None,
        internal_bc: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine a mesh using the red-green-blue refinement algorithm.

    Parameters:
        coordinates: np.ndarray
            Coordinates of the vertices. Shape is (num_vertices, 2).
        elements: np.ndarray
            Map between triangles and vertices. Shape is (num_elements, 3).
        marked_elements: np.ndarray, optional. Default is None.
            Boolean array of marked elements that should be refined.
            Shape is (num_elements, ).
        dirichlet_bc: np.ndarray, optional. Default is None.
            Array of Dirichlet boundary edges. Shape is (num_dirichlet_edges, 2).
        neumann_bc: np.ndarray, optional. Default is None.
            Array of Neumann boundary edges. Shape is (num_neumann_edges, 2).
        internal_bc: np.ndarray, optional. Default is None.
            Array of internal boundary edges. Shape is (num_internal_edges, 2).

    Returns:
        Five-tuple containing:
            - np.ndarray: Coordinates of the vertices of the refined grid.
            - np.ndarray: Mapping between triangles and vertices of the refined grid.
            - np.ndarray: Refined Dirichlet boundary edges.
            - np.ndarray: Refined Neumann boundary edges.
            - np.ndarray: Refined internal boundary edges.
    """

    numel = elements.shape[0]

    # Get all marked elements. If not given, assume uniform refinement
    if marked_elements is None:
        marked_elements = np.ones(numel, dtype=bool)

    # Retrieve edge-based geometric data
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements, dirichlet_bc, neumann_bc, internal_bc)

    # Prepare edge to new nodes map
    edge2new_node = np.zeros(np.max(element2edges) + 1, dtype=int)
    edge2new_node[element2edges[marked_elements].flatten()] = 1

    # Establish hash pattern for red-green-blue refinement
    hash_pattern = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    map_arr, value = hash_to_map(np.arange(8), hash_pattern)

    # Proceed to mark
    swap = [1]
    while len(swap) > 0:
        marked_edge = edge2new_node[element2edges]
        dec = np.sum(marked_edge[:numel] * (2 ** np.arange(3)), axis=1)
        val = value[dec]
        idx, jdx = np.where((marked_edge[:numel] == 0) & map_arr[dec])
        swap = idx + jdx * numel
        edge2new_node[element2edges[swap, 0]] = 1

    # Generate new nodes
    edge2new_node_indices = np.nonzero(edge2new_node)[0]
    new_node_indices = np.arange(
        len(coordinates), len(coordinates) + len(edge2new_node_indices)
    )
    edge2new_node[edge2new_node_indices] = new_node_indices

    coordinates = np.vstack(
        [
            coordinates,
            (
                    coordinates[edge2nodes[edge2new_node_indices][:, 0]] +
                    coordinates[edge2nodes[edge2new_node_indices][:, 1]]
            ) / 2
        ]
    )

    # Treatment of boundary conditions
    bc_list = []
    for i, boundary in enumerate([dirichlet_bc, neumann_bc, internal_bc]):
        if boundary is not None and len(boundary) > 0:
            new_nodes = edge2new_node[
                dir2edges if i == 0 else neu2edges if i == 1 else int2edges
            ].flatten()
            marked_edges = np.nonzero(new_nodes)[0]
            if len(marked_edges) > 0:
                boundary = np.vstack(
                    [boundary[new_nodes == 0],
                        np.column_stack(
                            [boundary[marked_edges, 0], new_nodes[marked_edges]]
                        ),
                        np.column_stack(
                            [new_nodes[marked_edges], boundary[marked_edges, 1]]
                        )
                    ]
                )
        bc_list.append(boundary)

    new_nodes = edge2new_node[element2edges]
    none = np.where(val == 0)[0]
    green = np.where(val == 1)[0]
    bluer = np.where(val == 2)[0]
    bluel = np.where(val == 3)[0]
    red = np.where(val == 4)[0]

    idx = np.ones(numel, dtype=int)
    idx[none] = 1
    idx[green] = 2
    idx[bluer] = 3
    idx[bluel] = 3
    idx[red] = 4
    idx = np.concatenate(([1], 1 + np.cumsum(idx)))

    new_elements = np.zeros((idx[-1] - 1, 3), dtype=int)
    new_elements[idx[none] - 1] = elements[none]

    tmp = np.column_stack([elements, new_nodes])
    new_elements[idx[green] - 1] = tmp[green][:, [2, 0, 3]]
    new_elements[idx[green]] = tmp[green][:, [1, 2, 3]]

    new_elements[idx[bluer] - 1] = tmp[bluer][:, [2, 0, 3]]
    new_elements[idx[bluer]] = tmp[bluer][:, [3, 1, 4]]
    new_elements[idx[bluer] + 1] = tmp[bluer][:, [2, 3, 4]]

    new_elements[idx[bluel] - 1] = tmp[bluel][:, [3, 2, 5]]
    new_elements[idx[bluel]] = tmp[bluel][:, [0, 3, 5]]
    new_elements[idx[bluel] + 1] = tmp[bluel][:, [1, 2, 3]]

    new_elements[idx[red] - 1] = tmp[red][:, [0, 3, 5]]
    new_elements[idx[red]] = tmp[red][:, [3, 1, 4]]
    new_elements[idx[red] + 1] = tmp[red][:, [5, 4, 2]]
    new_elements[idx[red] + 2] = tmp[red][:, [4, 5, 3]]

    return (coordinates, new_elements) + tuple(bc_list)
