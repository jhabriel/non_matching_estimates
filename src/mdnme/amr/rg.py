"""Red-Green refinement algorithm"""

import numpy as np
from mdnme.amr.refinement_utils import hash_to_map, get_edge_data_2d

# Global variable to track the number of green elements
nG = 0


def refine_rg(
        coordinates: np.ndarray,
        elements: np.ndarray,
        marked_elements: np.ndarray,
        dirichlet_bc: np.ndarray | None = None,
        neumann_bc: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine a mesh using the red-green refinement algorithm.

    Parameters:
        coordinates: np.ndarray
            Coordinates of the vertices. Shape is (num_vertices, 2).
        elements: np.ndarray
            Map between triangles and vertices. Shape is (num_elements, 3).
        marked_elements: np.ndarray
            Indices of elements to be refined.
        dirichlet_bc: np.ndarray, optional. Default is None.
            Array of Dirichlet boundary edges. Shape is (num_dirichlet_edges, 2).
        neumann_bc: np.ndarray, optional. Default is None.
            Array of Neumann boundary edges. Shape is (num_neumann_edges, 2).

    Returns:
        Tuple containing:
            - np.ndarray: Coordinates of the vertices of the refined grid.
            - np.ndarray: Mapping between triangles and vertices of the refined grid.
            - np.ndarray: Refined Dirichlet boundary edges.
            - np.ndarray: Refined Neumann boundary edges.
    """

    global nG
    num_elements = elements.shape[0]

    # Retrieve edge-based geometric data
    edge2nodes, element2edges, dir2edges, neu2edges, _ = get_edge_data_2d(elements,
                                                                          dirichlet_bc,
                                                                          neumann_bc)

    # Mark edges for refinement
    edge2new_node = np.zeros(np.max(element2edges) + 1, dtype=int)
    edge2new_node[element2edges[marked_elements].flatten()] = 1

    # Hash patterns for red and green refinement
    hash_red = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    map_red, value_red = hash_to_map(np.arange(8), hash_red)

    hash_green = np.array([
        [1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]
    ], dtype=bool)
    map_green, value_green = hash_to_map(np.arange(16), hash_green)

    # Proceed to mark
    swap = [1]
    while swap or np.any(edge2new_node[element2edges]):
        marked_edge = edge2new_node[element2edges]
        bit_red = marked_edge[:num_elements]
        dec_red = np.sum(bit_red * (2 ** np.arange(3)), axis=1)
        val_red = value_red[dec_red]
        idx, jdx = np.where((bit_red == 0) & map_red[dec_red])
        swap = idx + jdx * num_elements
        edge2new_node[element2edges[swap]] = 1

        bit_green = np.column_stack([
            marked_edge[num_elements // 2::2, 0:2],
            marked_edge[num_elements // 2 + 1::2, 0:2]
        ])
        dec_green = np.sum(bit_green * (2 ** np.arange(4)), axis=1)
        val_green = value_green[dec_green]
        green_idx = np.where(val_green)[0]
        flags = ~bit_green & map_green[dec_green]
        for k in range(4):
            edge2new_node[element2edges[num_elements // 2 + 2 * green_idx[flags[green_idx, k]] - 1, k // 2]] = 1

    # Generate new nodes
    new_node_indices = np.nonzero(edge2new_node)[0]
    new_node_idx = np.arange(
        len(coordinates), len(coordinates) + len(new_node_indices)
    )
    edge2new_node[new_node_indices] = new_node_idx

    coordinates = np.vstack(
        [
            coordinates,
            (
                    coordinates[edge2nodes[new_node_indices][:, 0]] +
                    coordinates[edge2nodes[new_node_indices][:, 1]]
            ) / 2
        ]
    )

    # Refine boundary conditions
    boundary_conditions = []
    for boundary, boundary_edges in zip(
            [dirichlet_bc, neumann_bc],
            [dir2edges, neu2edges]
    ):
        if boundary is not None:
            new_nodes = edge2new_node[boundary_edges].flatten()
            marked_edges = np.nonzero(new_nodes)[0]
            if len(marked_edges) > 0:
                boundary = np.vstack(
                    [boundary[new_nodes == 0],
                     np.column_stack([boundary[marked_edges, 0], new_nodes[marked_edges]]),
                     np.column_stack([new_nodes[marked_edges], boundary[marked_edges, 1]])
                     ]
                )
        boundary_conditions.append(boundary)

    # Refine elements
    new_nodes = edge2new_node[element2edges]
    none = np.where(val_red == 0)[0]
    r2red = np.where(val_red == 1)[0]
    r2green1 = np.where(val_red == 2)[0]
    r2green2 = np.where(val_red == 3)[0]
    r2green3 = np.where(val_red == 4)[0]

    g2green = np.where(val_green == 0)[0]
    g2red = np.where(val_green == 1)[0]
    g2red1 = np.where(val_green == 2)[0]
    g2red2 = np.where(val_green == 3)[0]
    g2red12 = np.where(val_green == 4)[0]

    rdx = np.zeros(num_elements, dtype=int)
    rdx[none] = 1
    rdx[np.concatenate([r2red, g2red])] = 4
    rdx[np.concatenate([g2red1, g2red2])] = 3
    rdx[g2red12] = 2
    rdx = np.concatenate(([1], 1 + np.cumsum(rdx)))

    gdx = np.zeros_like(rdx)
    gdx[np.concatenate([r2green1, r2green2, r2green3, g2green, g2red1, g2red2])] = 2
    gdx[g2red12] = 4
    gdx = rdx[-1] + np.concatenate(([0], np.cumsum(gdx)))

    new_elements = np.zeros((gdx[-1] - 1, 3), dtype=int)
    new_elements[rdx[none] - 1] = elements[none]

    tmp = np.column_stack([elements, new_nodes])

    # Assign new elements according to the refinement pattern
    new_elements[rdx[r2red] - 1] = tmp[r2red][:, [3, 1, 4]]
    new_elements[rdx[r2red]] = tmp[r2red][:, [0, 3, 4]]
    new_elements[rdx[r2red] + 1] = tmp[r2red][:, [4, 2, 0]]
    new_elements[rdx[r2red] + 2] = tmp[r2red][:, [1, 4, 3]]

    new_elements[rdx[g2red] - 1] = tmp[g2red][:, [3, 1, 4]]
    new_elements[rdx[g2red]] = tmp[g2red][:, [0, 3, 4]]
    new_elements[rdx[g2red] + 1] = tmp[g2red][:, [4, 2, 0]]
    new_elements[rdx[g2red] + 2] = tmp[g2red][:, [1, 4, 3]]

    new_elements[rdx[g2red1] - 1] = tmp[g2red1][:, [3, 1, 4]]
    new_elements[rdx[g2red1]] = tmp[g2red1][:, [4, 2, 0]]
    new_elements[rdx[g2red1] + 1] = tmp[g2red1][:, [1, 4, 3]]

    new_elements[rdx[g2red2] - 1] = tmp[g2red2][:, [3, 1, 4]]
    new_elements[rdx[g2red2]] = tmp[g2red2][:, [0, 3, 4]]
    new_elements[rdx[g2red2] + 1] = tmp[g2red2][:, [1, 4, 3]]

    new_elements[rdx[g2red12] - 1] = tmp[g2red12][:, [3, 1, 4]]
    new_elements[rdx[g2red12]] = tmp[g2red12][:, [0, 3, 4]]
    new_elements[rdx[g2red12] + 1] = tmp[g2red12][:, [1, 4, 3]]

    # New green elements
    new_elements[gdx[r2green1] - 1] = tmp[r2green1][:, [2, 0, 3]]
    new_elements[gdx[r2green1]] = tmp[r2green1][:, [3, 1, 2]]

    new_elements[gdx[r2green2] - 1] = tmp[r2green2][:, [0, 2, 4]]
    new_elements[gdx[r2green2]] = tmp[r2green2][:, [4, 3, 0]]

    new_elements[gdx[r2green3] - 1] = tmp[r2green3][:, [1, 3, 5]]
    new_elements[gdx[r2green3]] = tmp[r2green3][:, [5, 0, 1]]

    new_elements[gdx[g2green] - 1] = tmp[g2green][:, [2, 0, 3]]
    new_elements[gdx[g2green]] = tmp[g2green][:, [3, 1, 2]]

    new_elements[gdx[g2red1] - 1] = tmp[g2red1][:, [2, 0, 3]]
    new_elements[gdx[g2red1]] = tmp[g2red1][:, [3, 1, 2]]

    new_elements[gdx[g2red2] - 1] = tmp[g2red2][:, [0, 2, 4]]
    new_elements[gdx[g2red2]] = tmp[g2red2][:, [4, 3, 0]]

    new_elements[gdx[g2red12] - 1] = tmp[g2red12][:, [2, 0, 3]]
    new_elements[gdx[g2red12]] = tmp[g2red12][:, [3, 1, 2]]
    new_elements[gdx[g2red12] + 1] = tmp[g2red12][:, [0, 2, 4]]
    new_elements[gdx[g2red12] + 2] = tmp[g2red12][:, [4, 3, 0]]

    # Update nG
    nG = new_elements.shape[0] - rdx[-1] + 1

    return (coordinates, new_elements) + tuple(boundary_conditions)
