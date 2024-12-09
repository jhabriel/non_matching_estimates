"""
Module that tests the newest vertex bisection refinement algorithm.

Covered cases:
    T1: Single element.
        BCs: Dirichlet, Neumann, Internal, and Mixed.
    T2: Two elements bisecting a unit square.
        Marked elements: lower, upper, both.
        BCs: Dirichlet, Neumann, Internal, and Mixed.
    T3: Six elements gridding an L-shaped domain.
        Marked elements: Three out of six, arbitrarily selected.
        BCs: Dirichlet (top, bottom), Neumann (left, right), Internal (corner).
    T4: Two elements with duplicated middle nodes.
        Marked elements: left, right, both.

"""
from __future__ import annotations

import numpy as np
import pytest
from mdnme.amr.nvb import refine_nvb


class TestRefineNVBSingleCell:
    """Test class for NVB refinement with a single element."""

    def setup_method(self, method):
        self.coordinates = np.array([[0, 0], [1, 0], [0, 1]])
        self.elements = np.array([[0, 1, 2]])
        self.desired_new_coordinates = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.5, 0.5],
            ]
        )
        self.desired_new_elements = np.array(
            [
                [3, 2, 4],
                [0, 3, 4],
                [3, 1, 5],
                [2, 3, 5],
            ]
        )

    def assert_common(
            self,
            new_coordinates,
            new_elements,
            desired_new_coordinates,
            desired_new_elements
    ):
        np.testing.assert_equal(new_coordinates, desired_new_coordinates)
        np.testing.assert_equal(new_elements, desired_new_elements)

    def test_nvb_single_element(self):
        """Test nvb refinement for a single element without boundary conditions."""
        new_coordinates, new_elements, _, _, _ = refine_nvb(
            self.coordinates,
            self.elements
        )
        self.assert_common(
            new_coordinates,
            new_elements,
            self.desired_new_coordinates,
            self.desired_new_elements,
        )

    def test_nvb_single_element_dirichlet_bc(self):
        """Test nvb refinement for a single element with Dirichlet bcs."""
        dirichlet_bc = np.array([[0, 1], [1, 2], [2, 0]])
        marked_elements = np.array([0])
        new_coordinates, new_elements, new_dirichlet_bc, _, _ = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements,
            dirichlet_bc,
            None,
            None
        )
        desired_new_dirichlet_bc = np.array(
            [
                [0, 3],
                [1, 5],
                [2, 4],
                [3, 1],
                [5, 2],
                [4, 0],
            ]
        )
        self.assert_common(
            new_coordinates,
            new_elements,
            self.desired_new_coordinates,
            self.desired_new_elements
        )
        np.testing.assert_equal(new_dirichlet_bc, desired_new_dirichlet_bc)

    def test_nvb_single_element_neumann_bc(self):
        """Test nvb refinement for a single element with Neumann bcs."""
        neumann_bc = np.array([[0, 1], [1, 2], [2, 0]])
        marked_elements = np.array([0])
        new_coordinates, new_elements, _, new_neumann_bc, _ = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements,
            None,
            neumann_bc,
            None,
        )
        desired_new_neumann_bc = np.array(
            [
                [0, 3],
                [1, 5],
                [2, 4],
                [3, 1],
                [5, 2],
                [4, 0],
            ]
        )
        self.assert_common(
            new_coordinates,
            new_elements,
            self.desired_new_coordinates,
            self.desired_new_elements
        )
        np.testing.assert_equal(new_neumann_bc, desired_new_neumann_bc)

    def test_nvb_single_element_internal_bc(self):
        """Test nvb refinement for a single element with internal bcs."""
        internal_bc = np.array([[0, 1], [1, 2], [2, 0]])
        marked_elements = np.array([0])
        new_coordinates, new_elements, _, _, new_internal_bc = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements,
            None,
            None,
            internal_bc
        )
        desired_new_internal_bc = np.array(
            [
                [0, 3],
                [1, 5],
                [2, 4],
                [3, 1],
                [5, 2],
                [4, 0],
            ]
        )
        self.assert_common(
            new_coordinates,
            new_elements,
            self.desired_new_coordinates,
            self.desired_new_elements
        )
        np.testing.assert_equal(
            new_internal_bc,
            desired_new_internal_bc
        )

    def test_nvb_single_element_mixed_bc(self):
        """Test nvb refinement for a single element with mixed boundary conditions."""
        dirichlet_bc = np.array([[0, 1]])
        neumann_bc = np.array([[1, 2]])
        internal_bc = np.array([[2, 0]])
        marked_elements = np.array([0])
        new_coordinates, new_elements, new_dir_bc, new_neu_bc, new_int_bc = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements,
            dirichlet_bc,
            neumann_bc,
            internal_bc,
        )
        desired_new_dir_bc = np.array(
            [
                [0, 3],
                [3, 1],
            ]
        )
        desired_new_neu_bc = np.array(
            [
                [1, 5],
                [5, 2],
            ]
        )
        desired_new_int_bc = np.array(
            [
                [2, 4],
                [4, 0],
            ]
        )
        self.assert_common(
            new_coordinates,
            new_elements,
            self.desired_new_coordinates,
            self.desired_new_elements
        )
        np.testing.assert_equal(new_dir_bc, desired_new_dir_bc)
        np.testing.assert_equal(new_neu_bc, desired_new_neu_bc)
        np.testing.assert_equal(new_int_bc, desired_new_int_bc)


class TestRefineNVBTwoCells:
    """Test class for NVB refinement with two elements bisecting a unit square."""

    def setup_method(self, method):
        # Coordinates of the vertices of the original grid
        self.coordinates = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )
        # Triangles defined by the vertices of the original grid
        self.elements = np.array(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        # Expected new coordinates and elements for different marked elements
        # Scenarios:
        #   'lower_marked': The lower element is marked for refinement
        #   'upper_marked': The upper element is marked for refinement
        #   'both_marked': Both elements are marked for refinement
        self.scenarios = {
            'lower_marked': {
                'marked_elements': np.array([True, False]),
                'new_coordinates': np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [0.5, 0.0],
                        [0.0, 0.5],
                        [0.5, 0.5],
                        [1.0, 0.5],
                    ]
                ),
                'new_elements': np.array(
                    [
                        [4, 2, 5],
                        [0, 4, 5],
                        [4, 1, 6],
                        [2, 4, 6],
                        [7, 2, 6],
                        [1, 7, 6],
                        [3, 2, 7],
                    ]
                ),
                'new_bc_all_dir': np.array(
                    [
                        [3, 2],
                        [0, 4],
                        [1, 7],
                        [2, 5],
                        [4, 1],
                        [7, 3],
                        [5, 0],
                    ]
                ),
                'new_bc_all_neu': np.array(
                    [
                        [3, 2],
                        [0, 4],
                        [1, 7],
                        [2, 5],
                        [4, 1],
                        [7, 3],
                        [5, 0],
                    ]
                ),
                'new_bc_all_int': np.array(
                    [
                        [3, 2],
                        [0, 4],
                        [1, 7],
                        [2, 5],
                        [4, 1],
                        [7, 3],
                        [5, 0],
                    ]
                ),
                'new_bc_mixed_dir': np.array(
                    [
                        [0, 5],
                        [1, 7],
                        [5, 2],
                        [7, 3],
                    ]
                ),
                'new_bc_mixed_neu': np.array(
                    [
                        [2, 3],
                        [0, 4],
                        [4, 1],
                    ]
                ),
                'new_bc_mixed_int': np.array(
                    [
                        [1, 6],
                        [6, 2],
                    ]
                ),
            },
            'upper_marked': {
                'marked_elements': np.array([False, True]),
                'new_coordinates': np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [0.5, 0.0],
                        [0.5, 0.5],
                        [1.0, 0.5],
                        [0.5, 1.0],
                    ]
                ),
                'new_elements': np.array(
                    [
                        [2, 0, 4],
                        [4, 1, 5],
                        [2, 4, 5],
                        [6, 2, 5],
                        [1, 6, 5],
                        [6, 3, 7],
                        [2, 6, 7],
                    ]
                ),
                'new_bc_all_dir': np.array(
                    [
                        [2, 0],
                        [0, 4],
                        [1, 6],
                        [3, 7],
                        [4, 1],
                        [6, 3],
                        [7, 2],
                    ]
                ),
                'new_bc_all_neu': np.array(
                    [
                        [2, 0],
                        [0, 4],
                        [1, 6],
                        [3, 7],
                        [4, 1],
                        [6, 3],
                        [7, 2],
                    ]
                ),
                'new_bc_all_int': np.array(
                    [
                        [2, 0],
                        [0, 4],
                        [1, 6],
                        [3, 7],
                        [4, 1],
                        [6, 3],
                        [7, 2],
                    ]
                ),
                'new_bc_mixed_dir': np.array(
                    [
                        [0, 2],
                        [1, 6],
                        [6, 3],
                    ]
                ),
                'new_bc_mixed_neu': np.array(
                    [
                        [0, 4],
                        [2, 7],
                        [4, 1],
                        [7, 3],
                    ]
                ),
                'new_bc_mixed_int': np.array(
                    [
                        [1, 5],
                        [5, 2],
                    ]
                ),
            },
            'both_marked': {
                'marked_elements': np.array([True, True]),
                'new_coordinates': np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [0.5, 0.0],
                        [0.0, 0.5],
                        [0.5, 0.5],
                        [1.0, 0.5],
                        [0.5, 1.0],
                    ]
                ),
                'new_elements': np.array(
                    [
                        [4, 2, 5],
                        [0, 4, 5],
                        [4, 1, 6],
                        [2, 4, 6],
                        [7, 2, 6],
                        [1, 7, 6],
                        [7, 3, 8],
                        [2, 7, 8],
                    ]
                ),
                'new_bc_all_dir': np.array(
                    [
                        [0, 4],
                        [1, 7],
                        [3, 8],
                        [2, 5],
                        [4, 1],
                        [7, 3],
                        [8, 2],
                        [5, 0],
                    ]
                ),
                'new_bc_all_neu': np.array(
                    [
                        [0, 4],
                        [1, 7],
                        [3, 8],
                        [2, 5],
                        [4, 1],
                        [7, 3],
                        [8, 2],
                        [5, 0],
                    ]
                ),
                'new_bc_all_int': np.array(
                    [
                        [0, 4],
                        [1, 7],
                        [3, 8],
                        [2, 5],
                        [4, 1],
                        [7, 3],
                        [8, 2],
                        [5, 0],
                    ]
                ),
                'new_bc_mixed_dir': np.array(
                    [
                        [0, 5],
                        [1, 7],
                        [5, 2],
                        [7, 3],
                    ]
                ),
                'new_bc_mixed_neu': np.array(
                    [
                        [0, 4],
                        [2, 8],
                        [4, 1],
                        [8, 3],
                    ]
                ),
                'new_bc_mixed_int': np.array(
                    [
                        [1, 6],
                        [6, 2],
                    ]
                ),
            }
        }

    @pytest.mark.parametrize("scenario", ['lower_marked', 'upper_marked', 'both_marked'])
    def test_nvb_two_elements_without_boundary_conditions(self, scenario):
        """Parameterized test for nvb refinement with two elements."""
        scenario_data = self.scenarios[scenario]
        new_coordinates, new_elements, _, _, _ = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=scenario_data['marked_elements'],
            dirichlet_bc=None,
            neumann_bc=None,
            internal_bc=None,
        )
        np.testing.assert_equal(new_coordinates, scenario_data['new_coordinates'])
        np.testing.assert_equal(new_elements, scenario_data['new_elements'])

    @pytest.mark.parametrize("scenario", ['lower_marked', 'upper_marked', 'both_marked'])
    def test_nvb_two_elements_all_dirichlet(self, scenario):
        """Parameterized test for nvb refinement with two elements and Dirichlet bcs."""
        scenario_data = self.scenarios[scenario]
        new_coordinates, new_elements, new_dir_bc, _, _ = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=scenario_data['marked_elements'],
            dirichlet_bc=np.array([[0, 1], [1, 3], [3, 2], [2, 0]]),
            neumann_bc=None,
            internal_bc=None,
        )
        np.testing.assert_equal(new_coordinates, scenario_data['new_coordinates'])
        np.testing.assert_equal(new_elements, scenario_data['new_elements'])
        np.testing.assert_equal(new_dir_bc, scenario_data["new_bc_all_dir"])

    @pytest.mark.parametrize("scenario", ['lower_marked', 'upper_marked', 'both_marked'])
    def test_nvb_two_elements_all_neumann(self, scenario):
        """Parameterized test for nvb refinement with two elements and Neumann bcs."""
        scenario_data = self.scenarios[scenario]
        new_coordinates, new_elements, _, new_neu_bc, _ = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=scenario_data['marked_elements'],
            dirichlet_bc=None,
            neumann_bc=np.array([[0, 1], [1, 3], [3, 2], [2, 0]]),
            internal_bc=None,
        )
        np.testing.assert_equal(new_coordinates, scenario_data['new_coordinates'])
        np.testing.assert_equal(new_elements, scenario_data['new_elements'])
        np.testing.assert_equal(new_neu_bc, scenario_data["new_bc_all_neu"])

    @pytest.mark.parametrize("scenario",
                             ['lower_marked', 'upper_marked', 'both_marked'])
    def test_nvb_two_elements_all_internal(self, scenario):
        """Parameterized test for nvb refinement with two elements and internal bcs."""
        scenario_data = self.scenarios[scenario]
        new_coordinates, new_elements, _, _, new_int_bc = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=scenario_data['marked_elements'],
            dirichlet_bc=None,
            neumann_bc=None,
            internal_bc=np.array([[0, 1], [1, 3], [3, 2], [2, 0]]),
        )
        np.testing.assert_equal(new_coordinates, scenario_data['new_coordinates'])
        np.testing.assert_equal(new_elements, scenario_data['new_elements'])
        np.testing.assert_equal(new_int_bc, scenario_data["new_bc_all_int"])

    @pytest.mark.parametrize("scenario",
                             ['lower_marked', 'upper_marked', 'both_marked'])
    def test_nvb_two_elements_mixed_bc(self, scenario):
        """Parameterized test for nvb refinement with mixed bcs.

        Imposed boundary conditions on the original mesh:
            Left: Dirichlet
            Right: Dirichlet
            Bottom: Neumann
            Top: Neumann
            Diagonal: Internal

        """
        scenario_data = self.scenarios[scenario]
        new_coordinates, new_elements, new_dir_bc, new_neu_bc, new_int_bc = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=scenario_data['marked_elements'],
            dirichlet_bc=np.array([[0, 2], [1, 3]]),
            neumann_bc=np.array([[0, 1], [2, 3]]),
            internal_bc=np.array([[1, 2]]),
        )
        np.testing.assert_equal(new_coordinates, scenario_data['new_coordinates'])
        np.testing.assert_equal(new_elements, scenario_data['new_elements'])
        np.testing.assert_equal(new_dir_bc, scenario_data["new_bc_mixed_dir"])
        np.testing.assert_equal(new_neu_bc, scenario_data["new_bc_mixed_neu"])
        np.testing.assert_equal(new_int_bc, scenario_data["new_bc_mixed_int"])


class TestRefineLShapedSixCells:
    """
        Test refinement for an L-shaped domain with six cells.
    """
    def setup_method(self, method):

        # Vertices of the original grid
        self.coordinates = np.array(
            [
                [0, -1],
                [0, 0],
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, 1],
                [2, 1],
                [2, 0],
            ]
        )

        # Triangles of the original grid
        self.elements = np.array(
            [
                [0, 1, 2],
                [2, 1, 3],
                [1, 4, 2],
                [1, 5, 4],
                [1, 7, 5],
                [7, 6, 5],
            ]
        )

        # Dirichlet edges of the original grid
        self.dirichlet = np.array(
            [
                [2, 0],
                [4, 5],
                [5, 6],
            ]
        )

        # Neumann edges of the original grid
        self.neumann = np.array(
            [
                [4, 3],
                [3, 2],
                [6, 7],
            ]
        )

        # Internal edges of the original grid
        self.internal = np.array(
            [
                [0, 1],
                [1, 7],
            ]
        )

        # Marked elements
        self.marked_elements = np.array(
            [True, False, False, True, True, False]
        )

        # Cooridnates of the refined grid
        self.new_coordinates = np.array(
            [
                [0, -1],
                [0, 0],
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, 1],
                [2, 1],
                [2, 0],
                [0, -0.5],
                [-0.5, -1],
                [-0.5, -0.5],
                [-0.5, 0.5],
                [0, 0.5],
                [1, 0],
                [-0.5, 1],
                [1, 0.5],
                [2, 0.5]
            ]
        )

        # Elements of the refined grid
        self.new_elements = np.array(
            [
                [8, 2, 9],
                [0, 8, 9],
                [8, 1, 10],
                [2, 8, 10],
                [3, 2, 10],
                [1, 3, 10],
                [11, 2, 10],
                [1, 11, 10],
                [4, 2, 11],
                [12, 4, 11],
                [1, 12, 11],
                [12, 5, 14],
                [4, 12, 14],
                [13, 5, 12],
                [1, 13, 12],
                [13, 7, 15],
                [5, 13, 15],
                [16, 5, 15],
                [7, 16, 15],
                [6, 5, 16],
            ]
        )

        # New Dirichlet boundary conditions
        self.new_dir_bc = np.array(
            [
                [5, 6],
                [2, 9],
                [4, 14],
                [9, 0],
                [14, 5],
            ]
        )

        # New Neumann boundary conditions
        self.new_neu_bc = np.array(
            [
                [4, 3],
                [3, 2],
                [6, 16],
                [16, 7],
            ]
        )

        # New internal boundary conditions
        self.new_int_bc = np.array(
            [
                [0, 8],
                [1, 13],
                [8, 1],
                [13, 7],
            ]
        )

    def test_nvb_lshaped_six_elements(self):
        """Test coordinates, elements and boundary conditions for an L-shaped domain."""
        # Call refinement algorithm and retrieve refined data
        new_coo, new_ele, new_dir_bc, new_neu_bc, new_int_bc = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=self.marked_elements,
            dirichlet_bc=self.dirichlet,
            neumann_bc=self.neumann,
            internal_bc=self.internal,
        )
        # Assert actual vs. desired
        np.testing.assert_equal(new_coo, self.new_coordinates)
        np.testing.assert_equal(new_ele, self.new_elements)
        np.testing.assert_equal(new_dir_bc, self.new_dir_bc)
        np.testing.assert_equal(new_neu_bc, self.new_neu_bc)
        np.testing.assert_equal(new_int_bc, self.new_int_bc)


class TestRefineNVBTwoCellsWithRepeatedNodes:
    """Test class for NVB refinement with two elements with duplicated middle nodes.

    Note:
        The purpose of this test is to check whether the NVB algorithm is capable to
        deal with repeated nodes. This prepares the ground for dealing with internal
        boundaries in the context of mixed-dimensional grids.

    """
    def setup_method(self, method):
        # Coordinates of the original grid
        self.coordinates = np.array(
            [
                [0, -1],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 0],
                [0, 1],
            ]
        )
        # Vertices defining the triangles of the original grid
        self.elements = np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
            ]
        )
        # Scenarios to be tested:
        #   left_marked -> only the left element is marked for refinement
        #   right_marked -> only the right element is marked for refinement
        #   both_marked -> both elements are marked for refinement
        self.scenarios = {
            'left_marked': {
                'marked_elements': np.array([True, False]),
                'new_coordinates': np.array(
                    [
                        [0, -1],
                        [0, 1],
                        [-1, 0],
                        [0, -1],
                        [1, 0],
                        [0, 1],
                        [0, 0],
                        [-0.5, -0.5],
                        [-0.5, 0.5],
                    ]
                ),
                'new_elements': np.array(
                    [
                        [6, 2, 7],
                        [0, 6, 7],
                        [6, 1, 8],
                        [2, 6, 8],
                        [3, 4, 5],
                    ]
                ),
            },
            'right_marked': {
                'marked_elements': np.array([False, True]),
                'new_coordinates': np.array(
                    [
                        [0, -1],
                        [0, 1],
                        [-1, 0],
                        [0, -1],
                        [1, 0],
                        [0, 1],
                        [0.5, -0.5],
                        [0, 0],
                        [0.5, 0.5]
                    ]
                ),
                'new_elements': np.array(
                    [
                        [0, 1, 2],
                        [6, 5, 7],
                        [3, 6, 7],
                        [6, 4, 8],
                        [5, 6, 8],
                    ]
                ),
            },
            'both_marked': {
                'marked_elements': np.array([True, True]),
                'new_coordinates': np.array(
                    [
                        [0, -1],
                        [0, 1],
                        [-1, 0],
                        [0, -1],
                        [1, 0],
                        [0, 1],
                        [0, 0],
                        [-0.5, -0.5],
                        [-0.5, 0.5],
                        [0.5, -0.5],
                        [0, 0],
                        [0.5, 0.5]
                    ]
                ),
                'new_elements': np.array(
                    [
                        [6, 2, 7],
                        [0, 6, 7],
                        [6, 1, 8],
                        [2, 6, 8],
                        [9, 5, 10],
                        [3, 9, 10],
                        [9, 4, 11],
                        [5, 9, 11],
                    ]
                ),
            }
        }

    @pytest.mark.parametrize("scenario",
                             ['left_marked', 'right_marked', 'both_marked'])
    def test_nvb_two_elements_repeated_nodes_without_bc(self, scenario):
        """Parameterized test for nvb refinement with two elements."""
        scenario_data = self.scenarios[scenario]
        new_coordinates, new_elements, _, _, _ = refine_nvb(
            self.coordinates,
            self.elements,
            marked_elements=scenario_data['marked_elements'],
            dirichlet_bc=None,
            neumann_bc=None,
            internal_bc=None,
        )
        np.testing.assert_equal(new_coordinates, scenario_data['new_coordinates'])
        np.testing.assert_equal(new_elements, scenario_data['new_elements'])
