"""
This module contains hard-coded values of the errors associated to the coarser mesh
used in the first numerical example from the paper.
"""
from __future__ import annotations


import numpy as np
import porepy as pp
import pytest

import mdnme
from mdnme.examples.example_1.model import (
    VarelaJNum2023Setup,
    manu_incomp_fluid,
    manu_incomp_solid,
)
from mdnme.examples.example_1.true_errors import Varela2023JNumTrueErrors2d


def test_example_1_with_mesh_size_0125() -> None:
    """
    Test whether we obtain the majorant and true errors for the first example.

    """
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    material_constants = {"solid": solid_constants, "fluid": fluid_constants}
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.125},
    }

    # Run the model
    setup = VarelaJNum2023Setup(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    # Retrieve subdomains and data dictionaries
    sd_matrix, d_matrix = mdg.subdomains(return_data=True, dim=2)[0]
    sd_frac, d_frac = mdg.subdomains(return_data=True, dim=1)[0]
    intf, d_intf = mdg.interfaces(return_data=True, dim=1)[0]

    # Estimate errors
    mdnme.estimate_errors(mdg)  # Assume no sources
    d_matrix["estimates"]["residual_error"] = setup.exact_sol.residual_error_matrix(
        sd_matrix, d_matrix
    )
    d_frac["estimates"]["residual_error"] = setup.exact_sol.residual_error_fracture(
        sd_frac, d_frac
    )
    diffusive_sd_matrix = d_matrix["estimates"]["diffusive_error"]
    diffusive_sd_frac = d_frac["estimates"]["diffusive_error"]
    diffusive_intf_left = d_intf["estimates"]["diffusive_error"][
        int(intf.num_cells / 2) :
    ]
    diffusive_intf_right = d_intf["estimates"]["diffusive_error"][
        : int(intf.num_cells / 2)
    ]
    diffusive_error = (
        diffusive_sd_matrix.sum()
        + diffusive_sd_frac.sum()
        + diffusive_intf_left.sum()
        + diffusive_intf_right.sum()
    ) ** 0.5
    residual_sd_matrix = d_matrix["estimates"]["residual_error"]
    residual_sd_frac = d_frac["estimates"]["residual_error"]
    residual_error = (residual_sd_matrix.sum() + residual_sd_frac.sum()) ** 0.5

    # Majorant, true error, and efficiency index
    majorant = diffusive_error + residual_error
    te = Varela2023JNumTrueErrors2d()
    true_error = te.true_error(mdg)
    efficiency_index = majorant / true_error

    # Check against known values
    np.testing.assert_allclose(majorant, 0.11336778184222394)
    np.testing.assert_allclose(true_error, 0.10057203148161831)
    np.testing.assert_allclose(efficiency_index, 1.1272297096130979)


def test_example_1_with_mesh_size_0125_nonmatching() -> None:
    """
    Test whether we obtain the majorant and true errors for the first example. The mdg
    is fully matching, although it is created using the non-matching machinery.

    """
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    material_constants = {"solid": solid_constants, "fluid": fluid_constants}
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "non_matching_cell_sizes": (0.125, 0.125, 0.125),
    }

    # Run the model
    setup = VarelaJNum2023Setup(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    # Retrieve subdomains and data dictionaries
    sd_matrix, d_matrix = mdg.subdomains(return_data=True, dim=2)[0]
    sd_frac, d_frac = mdg.subdomains(return_data=True, dim=1)[0]
    intf, d_intf = mdg.interfaces(return_data=True, dim=1)[0]

    # Estimate errors
    mdnme.estimate_errors(mdg)  # Assume no sources
    d_matrix["estimates"]["residual_error"] = setup.exact_sol.residual_error_matrix(
        sd_matrix, d_matrix
    )
    d_frac["estimates"]["residual_error"] = setup.exact_sol.residual_error_fracture(
        sd_frac, d_frac
    )
    diffusive_sd_matrix = d_matrix["estimates"]["diffusive_error"]
    diffusive_sd_frac = d_frac["estimates"]["diffusive_error"]
    diffusive_intf_left = d_intf["estimates"]["diffusive_error"][
        int(intf.num_cells / 2) :
    ]
    diffusive_intf_right = d_intf["estimates"]["diffusive_error"][
        : int(intf.num_cells / 2)
    ]
    diffusive_error = (
        diffusive_sd_matrix.sum()
        + diffusive_sd_frac.sum()
        + diffusive_intf_left.sum()
        + diffusive_intf_right.sum()
    ) ** 0.5
    residual_sd_matrix = d_matrix["estimates"]["residual_error"]
    residual_sd_frac = d_frac["estimates"]["residual_error"]
    residual_error = (residual_sd_matrix.sum() + residual_sd_frac.sum()) ** 0.5

    # Majorant, true error, and efficiency index
    majorant = diffusive_error + residual_error
    te = Varela2023JNumTrueErrors2d()
    true_error = te.true_error(mdg)
    efficiency_index = majorant / true_error

    # Check against known values
    np.testing.assert_allclose(majorant, 0.11336778184222394)
    np.testing.assert_allclose(true_error, 0.10057203148161831)
    np.testing.assert_allclose(efficiency_index, 1.1272297096130979)



@pytest.mark.parametrize(
    "intbound_cell_size, interface_cell_size, fracture_cell_size",
    [
        (0.25, 0.125, 0.1),
        (0.25, 0.1, 0.125),
        (0.125, 0.25, 0.1),
        (0.125, 0.1, 0.25),
        (0.1, 0.25, 0.125),
        (0.1, 0.125, 0.25),
    ],
)
def test_non_matching_grids(intbound_cell_size, interface_cell_size, fracture_cell_size):
    """Checks whether non-matching grids are correctly generated by the model."""
    coupling_triplet_cell_size = (
        intbound_cell_size, interface_cell_size, fracture_cell_size
    )

    # Define material constants
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    material_constants = {"solid": solid_constants, "fluid": fluid_constants}

    # Setup simulation parameters
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "non_matching_cell_sizes": coupling_triplet_cell_size,
    }
    setup = VarelaJNum2023Setup(params)
    setup.prepare_simulation()
    mdg = setup.mdg

    # Assert cell sizes of internal boundary cells
    sd_matrix = mdg.subdomains()[0]
    frac_faces = sd_matrix.tags["fracture_faces"]
    np.testing.assert_array_almost_equal(
        sd_matrix.face_areas[frac_faces], intbound_cell_size
    )

    # Assert cell sizes of fracture cells
    sd_fracture = mdg.subdomains()[1]
    np.testing.assert_array_almost_equal(
        sd_fracture.cell_volumes, fracture_cell_size
    )

    # Assert cell sizes of interface cells
    intf = mdg.interfaces()[0]
    np.testing.assert_array_almost_equal(
        intf.cell_volumes, interface_cell_size
    )




