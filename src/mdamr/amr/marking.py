"""
Module containing functionality for marking elments to be refined.
"""

import numpy as np


def doerfler_marking(error_indicators: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply Dörfler marking strategy to mark elements for refinement.

    Parameters:
    - error_indicators: np.ndarray
        Array of error indicators for each element.
    - theta: float
        Threshold fraction of the total error. Should be between 0 and 1.
        This indicates the fraction of the total error that the marked elements
        should contribute to.

    Returns:
    - np.ndarray:
        Boolean array where True indicates the element is marked for refinement.

    """
    # Sort the error indicators in descending order and get the indices
    sorted_indices = np.argsort(-error_indicators)
    sorted_errors = error_indicators[sorted_indices]

    # Compute the cumulative sum of the sorted errors
    cumulative_errors = np.cumsum(sorted_errors)

    # Compute the total error
    total_error = cumulative_errors[-1]

    # Find the min number of elements whose cumulative error exceeds theta * total_error
    mark_threshold = theta * total_error
    num_elements_to_mark = np.searchsorted(cumulative_errors, mark_threshold) + 1

    # Mark these elements in the original array
    marked_elements = np.zeros_like(error_indicators, dtype=bool)
    marked_elements[sorted_indices[:num_elements_to_mark]] = True

    return marked_elements
