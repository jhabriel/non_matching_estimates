"""
Module containing refinement algorithms in 1d.
"""

import numpy as np


def refine_red_1d(
        coordinates: np.ndarray,
        elements: np.ndarray,
        marked_elements: np.ndarray = None
        ):
    """
    Perform red-refinement on a 1D mesh by subdividing marked intervals.

    Parameters:
        coordinates: np.ndarray
            Coordinates of the vertices in the 1D mesh. Shape is (num_vertices, 1).
        elements: np.ndarray
            Array defining the intervals (edges) by their vertices.
            Shape is (num_elements, 2).
        marked_elements: np.ndarray, optional. Default is None.
            Boolean array of marked elements that should be refined.
            Shape is (num_elements, ).

    Returns:
        tuple: (new_coordinates, new_elements)
            - new_coordinates: Coordinates of the vertices after refinement.
            - new_elements: Array of the refined intervals (edges).
    """

    # Get number of elements (intervals)
    num_elements = elements.shape[0]

    # If no marked elements provided, refine all elements
    if marked_elements is None:
        marked_elements = np.ones(num_elements, dtype=bool)

    # List to store new coordinates and elements
    new_coordinates = list(coordinates)
    new_elements = []

    # Keep track of newly added midpoints
    midpoint_map = {}

    # Iterate over all elements (intervals)
    for i, elem in enumerate(elements):
        # Get the two vertices of the current interval
        v1, v2 = elem

        if marked_elements[i]:
            # If the element is marked for refinement, insert the midpoint
            if (v1, v2) not in midpoint_map:
                # Calculate the midpoint
                midpoint = (coordinates[v1] + coordinates[v2]) / 2.0

                # Add the midpoint to the list of coordinates
                new_coordinates.append(midpoint)

                # Store the index of the new midpoint
                midpoint_index = len(new_coordinates) - 1
                midpoint_map[(v1, v2)] = midpoint_index
            else:
                midpoint_index = midpoint_map[(v1, v2)]

            # Create two new intervals
            new_elements.append([v1, midpoint_index])
            new_elements.append([midpoint_index, v2])
        else:
            # If not marked, just copy the original interval
            new_elements.append([v1, v2])

    # Convert the new coordinates and elements to numpy arrays
    new_coordinates = np.array(new_coordinates)
    new_elements = np.array(new_elements)

    return new_coordinates, new_elements
