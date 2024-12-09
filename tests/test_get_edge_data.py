"""Module to test edge-based data geometric information."""
from __future__ import annotations

import numpy as np
from mdnme.amr.refinement_utils import get_edge_data_2d


def test_single_triangle_no_extra_args():
    """Check edge geometric information for a single triangle."""
    elements = np.array([[0, 1, 2]])
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements)

    # Check shapes
    assert edge2nodes.shape == (3, 2)
    assert element2edges.shape == (1, 3)
    assert dir2edges.size == 0 and dir2edges.shape == (0, 1)
    assert neu2edges.size == 0 and neu2edges.shape == (0, 1)
    assert int2edges.size == 0 and int2edges.shape == (0, 1)

    # Check values
    np.testing.assert_equal(edge2nodes, np.array([[0, 1], [0, 2], [1, 2]]))
    np.testing.assert_equal(element2edges, np.array([[0, 2, 1]]))


def test_single_triangle_with_dirichlet():
    """Check edge geometric information for a single triangle with Dirichlet bc data."""
    elements = np.array([[0, 1, 2]])
    dirichlet = np.array([[0, 1], [1, 2], [2, 0]])
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements, dirichlet, None, None)

    # Check shapes
    assert edge2nodes.shape == (3, 2)
    assert element2edges.shape == (1, 3)
    assert dir2edges.size == 3 and dir2edges.shape == (3, 1)
    assert neu2edges.size == 0 and neu2edges.shape == (0, 1)
    assert int2edges.size == 0 and int2edges.shape == (0, 1)

    # Check values
    desired_edge2nodes = np.array([[0, 1], [0, 2], [1, 2]])
    desired_element2edges = np.array([[0, 2, 1]])
    desired_dir2edges = np.array([[0], [2], [1]])
    np.testing.assert_equal(edge2nodes, desired_edge2nodes)
    np.testing.assert_equal(element2edges, desired_element2edges)
    np.testing.assert_equal(dir2edges, desired_dir2edges)


def test_single_triangle_with_neumann():
    """Check edge geometric information for a single triangle with Neumann bc data."""
    elements = np.array([[0, 1, 2]])
    neumann = np.array([[0, 1], [1, 2], [2, 0]])
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements, None, neumann, None)

    # Check shapes
    assert edge2nodes.shape == (3, 2)
    assert element2edges.shape == (1, 3)
    assert dir2edges.size == 0 and dir2edges.shape == (0, 1)
    assert neu2edges.size == 3 and neu2edges.shape == (3, 1)
    assert int2edges.size == 0 and int2edges.shape == (0, 1)

    # Check values
    desired_edge2nodes = np.array([[0, 1], [0, 2], [1, 2]])
    desired_element2edges = np.array([[0, 2, 1]])
    desired_neu2edges = np.array([[0], [2], [1]])
    np.testing.assert_equal(edge2nodes, desired_edge2nodes)
    np.testing.assert_equal(element2edges, desired_element2edges)
    np.testing.assert_equal(neu2edges, desired_neu2edges)


def test_single_triangle_with_internal():
    """Check edge geometric information for a single triangle with internal bc data.

    Note: The function `get_edge_data` "does not know" whether an edge is truly
    internal or external. This test is mainly to catch any gross inconsistency.

    """
    elements = np.array([[0, 1, 2]])
    internal = np.array([[0, 1], [1, 2], [2, 0]])
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements, None, None, internal)

    # Check shapes
    assert edge2nodes.shape == (3, 2)
    assert element2edges.shape == (1, 3)
    assert dir2edges.size == 0 and dir2edges.shape == (0, 1)
    assert neu2edges.size == 0 and neu2edges.shape == (0, 1)
    assert int2edges.size == 3 and int2edges.shape == (3, 1)

    # Check values
    desired_edge2nodes = np.array([[0, 1], [0, 2], [1, 2]])
    desired_element2edges = np.array([[0, 2, 1]])
    desired_int2edges = np.array([[0], [2], [1]])
    np.testing.assert_equal(edge2nodes, desired_edge2nodes)
    np.testing.assert_equal(element2edges, desired_element2edges)
    np.testing.assert_equal(int2edges, desired_int2edges)


def test_single_triangle_mixed_bc():
    """Check edge geometric information for a single triangle with mixed bc's.

        Note: The function `get_edge_data` "does not know" whether an edge is truly
        internal or external. This test is mainly to catch any gross inconsistency.

    """
    elements = np.array([[0, 1, 2]])
    dirichlet = np.array([[0, 1]])
    neumann = np.array([[1, 2]])
    internal = np.array([[2, 0]])
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements, dirichlet, neumann, internal)

    # Check shapes
    assert edge2nodes.shape == (3, 2)
    assert element2edges.shape == (1, 3)
    assert dir2edges.size == 1 and dir2edges.shape == (1, 1)
    assert neu2edges.size == 1 and neu2edges.shape == (1, 1)
    assert int2edges.size == 1 and int2edges.shape == (1, 1)

    # Check values
    desired_edge2nodes = np.array([[0, 1], [0, 2], [1, 2]])
    desired_element2edges = np.array([[0, 2, 1]])
    desired_dir2edges = np.array([[0]])
    desired_neu2edges = np.array([[2]])
    desired_int2edges = np.array([[1]])
    np.testing.assert_equal(edge2nodes, desired_edge2nodes)
    np.testing.assert_equal(element2edges, desired_element2edges)
    np.testing.assert_equal(dir2edges, desired_dir2edges)
    np.testing.assert_equal(neu2edges, desired_neu2edges)
    np.testing.assert_equal(int2edges, desired_int2edges)


def test_unit_square_two_triangles_no_extra_args():
    """Test unit square with two triangles without boundary conditions"""
    elements = np.array([[0, 1, 2], [2, 1, 3]])
    edge2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements)

    # Check shapes
    assert edge2nodes.shape == (5, 2)
    assert element2edges.shape == (2, 3)
    assert dir2edges.size == 0 and dir2edges.shape == (0, 1)
    assert neu2edges.size == 0 and neu2edges.shape == (0, 1)
    assert int2edges.size == 0 and int2edges.shape == (0, 1)

    # Check values
    desired_edge2nodes = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    desired_element2edges = np.array([[0, 2, 1], [2, 3, 4]])
    np.testing.assert_equal(edge2nodes, desired_edge2nodes)
    np.testing.assert_equal(element2edges, desired_element2edges)


def test_unit_square_two_triangles_mixed_bc():
    """Test unit square with two triangles and mixed boundary conditions."""
    elements = np.array([[0, 1, 2], [2, 1, 3]])
    dirichlet = np.array([[0, 1], [2, 3]])
    neumann = np.array([[1, 3], [0, 2]])
    internal = np.array([[1, 2]])
    edges2nodes, element2edges, dir2edges, neu2edges, int2edges = get_edge_data_2d(
        elements, dirichlet, neumann, internal)

    # Check shapes
    assert edges2nodes.shape == (5, 2)
    assert element2edges.shape == (2, 3)
    assert dir2edges.size == 2 and dir2edges.shape == (2, 1)
    assert neu2edges.size == 2 and neu2edges.shape == (2, 1)
    assert int2edges.size == 1 and int2edges.shape == (1, 1)

    # Check values
    desired_edge2nodes = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    desired_element2edges = np.array([[0, 2, 1], [2, 3, 4]])
    desired_dir2edges = np.array([[0], [4]])
    desired_neu2edges = np.array([[3], [1]])
    desired_int2edges = np.array([[2]])

    np.testing.assert_equal(edges2nodes, desired_edge2nodes)
    np.testing.assert_equal(element2edges, desired_element2edges)
    np.testing.assert_equal(dir2edges, desired_dir2edges)
    np.testing.assert_equal(neu2edges, desired_neu2edges)
    np.testing.assert_equal(int2edges, desired_int2edges)
