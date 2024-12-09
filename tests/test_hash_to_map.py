"""
Module containing unit tests for the hash_to_map function
"""
from __future__ import annotations

import numpy as np

from mdnme.amr.refinement_utils import hash_to_map


def test_nvb_map_2d():
    """Test values and hash map for newest vertex bisect (nvb) in two dimensions."""

    # Hash patterns and possible arrangements for nvb
    hash_pattern = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    possible_arrangements = np.arange(8)

    # Get hash map and values
    hash_map, value = hash_to_map(possible_arrangements, hash_pattern)

    # Desired hash maps and values
    desired_hash_map = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    )
    desired_values = np.array([0, 1, 2, 2, 3, 3, 4, 4])

    # Test
    np.testing.assert_equal(hash_map, desired_hash_map)
    np.testing.assert_equal(value, desired_values)


def test_red_refinement_map_2d():
    """Test values and hash map for red refinement in two dimensions."""

    # Hash patterns and possible arrangements for red refinement
    hash_pattern = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    possible_arrangements = np.arange(8)

    # Get hash map and values
    hash_map, value = hash_to_map(possible_arrangements, hash_pattern)

    # Desired hash maps and values
    desired_hash_map = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    )
    desired_values = np.array([0, 2, 3, 1, 4, 1, 1, 1])

    # Test
    np.testing.assert_equal(hash_map, desired_hash_map)
    np.testing.assert_equal(value, desired_values)


def test_green_refinement_map_2d():
    """Test values and hash map for green refinement in two dimensions."""

    # Hash patterns and possible arrangements for green refinement
    hash_pattern = np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    possible_arrangements = np.arange(16)

    # Get hash map and values
    hash_map, value = hash_to_map(possible_arrangements, hash_pattern)

    # Desired hash maps and values
    desired_hash_map = np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 1],
         [1, 1, 0, 1],
         [1, 1, 0, 1],
         [1, 0, 1, 1],
         [1, 0, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [1, 1, 0, 1],
         [1, 1, 0, 1],
         [1, 0, 1, 1],
         [1, 0, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    )
    desired_values = np.array(
        [0, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4]
    )

    # Test
    np.testing.assert_equal(hash_map, desired_hash_map)
    np.testing.assert_equal(value, desired_values)
