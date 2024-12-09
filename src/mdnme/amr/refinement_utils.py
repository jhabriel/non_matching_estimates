"""
Module containing utility functions for the adaptive mesh refinement process.
"""

import numpy as np
import porepy as pp
import warnings
import matplotlib.pyplot as plt

from typing import Optional


def get_face_data(elements: np.ndarray,
                  dirichlet_bc: np.ndarray | None = None,
                  neumann_bc: np.ndarray | None = None,
                  internal_bc: np.ndarray | None = None) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get face-based geometric data for a 3D mesh.

    Parameters:
        elements: np.ndarray
            Tetrahedral elements with shape (num_elements, 4).
        dirichlet_bc: np.ndarray, optional. Default is None.
            Array of Dirichlet boundary faces. Shape is (num_dirichlet_faces, 3).
        neumann_bc: np.ndarray, optional. Default is None.
            Array of Neumann boundary faces. Shape is (num_neumann_faces, 3).
        internal_bc: np.ndarray, optional. Default is None.
            Array of internal boundary faces. Shape is (num_internal_faces, 3).

    Returns:
        tuple of np.ndarray:
            - face2nodes: Map from faces to their vertex nodes. Shape (num_faces, 3).
            - element2faces: Map from tetrahedrons to their faces. Shape (num_elements, 4).
            - dir2faces: Map from Dirichlet faces to their vertex nodes. Shape (num_dirichlet_faces, 3).
            - neu2faces: Map from Neumann faces to their vertex nodes. Shape (num_neumann_faces, 3).
            - int2faces: Map from internal faces to their vertex nodes. Shape (num_internal_faces, 3).
    """

    # Define faces for each tetrahedron
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])

    # Map from element to faces
    element2faces = np.array([np.sort(elements[:, faces[i]], axis=1) for i in range(4)])
    element2faces = element2faces.transpose(1, 0, 2)

    # Sort each face's nodes to prevent duplicates
    element2faces = np.sort(element2faces, axis=2)

    # Hash faces for unique identification
    unique_faces, face_indices = np.unique(element2faces.reshape(-1, 3), axis=0,
                                           return_inverse=True)

    # Reshape to element2faces format
    element2faces = face_indices.reshape(elements.shape[0], 4)

    # Map faces for boundary conditions
    dir2faces = None
    neu2faces = None
    int2faces = None

    if dirichlet_bc is not None:
        dir2faces = np.sort(dirichlet_bc, axis=1)
        dir2faces = np.searchsorted(unique_faces, dir2faces, axis=0)

    if neumann_bc is not None:
        neu2faces = np.sort(neumann_bc, axis=1)
        neu2faces = np.searchsorted(unique_faces, neu2faces, axis=0)

    if internal_bc is not None:
        int2faces = np.sort(internal_bc, axis=1)
        int2faces = np.searchsorted(unique_faces, int2faces, axis=0)

    return unique_faces, element2faces, dir2faces, neu2faces, int2faces


def get_edge_data(
        elements: np.ndarray,
        dirichlet: Optional[np.ndarray] = None,
        neumann: Optional[np.ndarray] = None,
        internal: Optional[np.ndarray] = None,
):
    """Wrapper to get edge data from 2D and 3D grids.

    :param elements:
    :param dirichlet:
    :param neumann:
    :param internal:
    :return:
    """
    if elements.shape[1] == 4:
        edge2nodes, element2edges, dir2edges, neu2edges, int2edges = (
            get_edge_data_3d(elements, dirichlet, neumann, internal)
        )
    elif elements.shape[1] == 3:
        edge2nodes, element2edges, dir2edges, neu2edges, int2edges = (
            get_edge_data_2d(elements, dirichlet, neumann, internal)
        )
    else:
        raise ValueError("Expected 2d or 3d FEM grid to employ this functionality.")

    return edge2nodes, element2edges, dir2edges, neu2edges, int2edges


def get_edge_data_3d(
    elements: np.ndarray,
    dirichlet: Optional[np.ndarray] = None,
    neumann: Optional[np.ndarray] = None,
    internal: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


def get_edge_data_2d(
        elements: np.ndarray,
        dirichlet: Optional[np.ndarray] = None,
        neumann: Optional[np.ndarray] = None,
        internal: Optional[np.ndarray] = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provides edge-based geometric data.

    Credits:
        This script has been adapted from [1], which is provided under MIT License.

    Parameters:
        elements: np.ndarray
            Triangles defined by the vertices. It is assumed that each triangle is
            positively given in a mathematical sense. Shape of array is
            (number_of_elements, 3).
        dirichlet: np.ndarray | None, default is None
            If given, it defines the Dirichlet edges using the vertices. Size is
            (num_dirichlet_edges, 2).
        neumann: np.ndarray | None, default is None
        internal: np.ndarray | None, default is None

    Raises:
        ValueError
            - If *args is different from `dirichlet`, `neumann` or `internal`.

    Returns:
        Three-tuple containing the following elements:

            np.ndarray
                Edge to nodes mapping. Shape is (number_of_edges, 2).
            np.ndarray
                Element to edges mapping. Shape is (number of elements, 3).
            list
                Mappings between boundary conditions and edges. The list can contain
                upmost three elements, corresponding respectively to dirichlet,
                neumann, and internal boundary conditions, each of which of shape
                (num_bc_type_edge, 1). The list is empty if *args is not given. The
                ordering of the elements in the list is the same as the one given in
                *args.

    References:
        [1] Funken, S. A., & Schmidt, A. (2020). Adaptive mesh refinement in 2D: An
            efficient implementation in matlab. Computational methods in applied
            mathematics, 20(3), 459-479.

    """
    if dirichlet is None:
        dirichlet = np.empty(shape=(0, 2), dtype=np.int32)
    if neumann is None:
        neumann = np.empty(shape=(0, 2), dtype=np.int32)
    if internal is None:
        internal = np.empty(shape=(0, 2), dtype=np.int32)

    # Collect boundary data in a list
    bc_data = [dirichlet, neumann, internal]

    # Collect all edges
    edges = elements[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)

    # Combine with additional boundary edges if provided
    ptr = [3 * elements.shape[0]]
    for bc in bc_data:
        ptr.append(bc.shape[0])
        edges = np.vstack([edges, bc])
    ptr = np.cumsum(ptr)

    # Sort edges to ensure consistent direction and find unique edges
    edges_sorted = np.sort(edges, axis=1)
    edge2nodes, unique_indices, ie = np.unique(
        edges_sorted, axis=0, return_index=True, return_inverse=True
    )

    # Relate edges to elements
    element2edges = ie[:ptr[0]].reshape(-1, 3)

    # Relate boundary edges to boundary conditions
    bc_list_to_edges = []
    for i, bc in enumerate(bc_data):
        bc_list_to_edges.append(ie[ptr[i]:ptr[i+1]].reshape(-1, 1))
    dir2edges = bc_list_to_edges[0]
    neu2edges = bc_list_to_edges[1]
    int2edges = bc_list_to_edges[2]

    return edge2nodes, element2edges, dir2edges, neu2edges, int2edges


def hash_to_map(
        dec: np.ndarray,
        hash_patterns: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    """ hash2map: maps markings given in decimal number to given hashes

    Usage:
    map, val = hash2map(dec, hash)

    Comments:
    hash2map expects as input a decimal number which determines the marking
    uniquely (e.g. 110 corresponds to dec = 3) and a hash where all possible
    refinement patterns are described. The function maps the marking to one
    of the edges by finding the pattern for which at least further edges have
    to be marked.

    The function returns the mapping and an assigned val for the hash.

    Remark:
    This program is a supplement to the paper
    >> Adaptive Mesh Refinement in 2D - An Efficient Implementation in Matlab <<
    by S. Funken, and A. Schmidt. The reader should
    consult that paper for more information.

    Authors:
    S. Funken, A. Schmidt  20-08-18
    """

    n = hash_patterns.shape[1]
    bin_arr = ((dec[:, None] * (2 ** np.arange(1-n, 1, dtype=float))) % 2).astype(int)
    bin_arr = np.fliplr(bin_arr)
    map_arr = np.zeros_like(bin_arr)
    val = np.zeros(dec.shape, dtype=int)

    for i in range(bin_arr.shape[0]):
        if dec[i]:
            # Compute the Hamming distance between the current
            # bin_arr row and each row in hash
            hamming_dist = np.sum(np.abs(hash_patterns - bin_arr[i]), axis=1)
            # Find the index of the row in hash with the minimum Hamming distance
            mdx = np.argmin(hamming_dist)
            map_arr[i] = hash_patterns[mdx]
            val[i] = mdx + 1  # Adding 1 to match the expected val format

    return map_arr, val


def enforce_positive_orientation(
        coordinates: np.ndarray,
        triangles: np.ndarray,
        ) -> np.ndarray:
    """Ensure all triangles are positively oriented

    Raises a warning if any triangles
    are not positively oriented before enforcing the orientation.

    Parameters:
    - coordinates: A num_coordinates x 2 numpy array containing the coordinates of the nodes.
    - triangles: A num_triangles x 3 numpy array containing the indices of the nodes forming each triangle.

    Returns:
    - A new array of triangles where all triangles are positively oriented.
    """

    def is_positively_oriented(x1, y1, x2, y2, x3, y3):
        return (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) > 0

    # Make a copy of the triangles to avoid modifying the input array directly
    new_triangles = np.copy(triangles)

    for i, triangle in enumerate(triangles):
        # Extract the coordinates of the nodes for the current triangle
        x1, y1 = coordinates[triangle[0]]
        x2, y2 = coordinates[triangle[1]]
        x3, y3 = coordinates[triangle[2]]

        # Check if the triangle is positively oriented
        if not is_positively_oriented(x1, y1, x2, y2, x3, y3):
            # If we find a negatively oriented triangle, flag it
            found_negative_orientation = True
            # Swap the first two vertices to enforce positive orientation
            new_triangles[i, [0, 1]] = new_triangles[i, [1, 0]]
        else:
            found_negative_orientation = False

        # Raise a warning if any negative orientations were found
        if found_negative_orientation:
            # msg = "Found at least one simplex whose nodes are not positively"
            # msg += " oriented. Enforcing positive orientation."
            # warnings.warn(msg, UserWarning)
            print("Found negative orientation")

    return new_triangles


def plot_fem_mesh(
        coordinates: np.ndarray,
        triangles: np.ndarray,
        filename='fem_mesh.png'
        ) -> None:
    """Plot the mesh corresponding to the given coordinates and elements.

    This function save the plot as a PNG image in the current directory.

    Parameters:
    - coordinates: A num_coordinates x 2 numpy array containing the coordinates of the nodes.
    - triangles: A num_triangles x 3 numpy array containing the indices of the nodes forming each triangle.
    - filename: Optional; The name of the file to save the image as. Defaults to 'fem_mesh.png'.

    Returns:
    - None. The function saves the mesh plot as an image.
    """
    plt.figure(figsize=(8, 8))

    for triangle in triangles:
        # Get the vertex coordinates for the triangle
        pts = coordinates[triangle]
        # Close the triangle by repeating the first point at the end
        pts = np.vstack([pts, pts[0]])
        # Plot the triangle
        plt.plot(pts[:, 0], pts[:, 1], 'bo-', linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mesh Plot')
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(filename, dpi=300)
    plt.close()


def random_marking(elements, seed=42):
    """
    Generate a random boolean array based on the size of the elements array.

    Parameters:
    - elements: A num_triangles x 3 numpy array containing the indices of the nodes forming each triangle.
    - seed: Optional; An integer seed for the random number generator. Defaults to 42.

    Returns:
    - A numpy array of boolean values of size equal to the number of triangles.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    num_triangles = elements.shape[0]
    return np.random.choice([True, False], size=num_triangles)
