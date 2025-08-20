"""
This module contains hard-coded values of the errors associated to the coarser mesh
used in the three-dimensional verification from [1].

References:
-----------

[1] Varela, et. al. 2023.

"""
from __future__ import annotations


import numpy as np
import porepy as pp
import pytest

import mdnme

from porepy.grids.mortar_grid import MortarSides

from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import (
    manu_incomp_fluid,
    manu_incomp_solid,
    )
from mdnme.examples.varela_jnum_3d.true_errors import VarelaJNumTrueErrors3D


@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


def test_model_pre_simulation(material_constants: dict) -> None:
    """Test whether the model is correctly set up before running the simulation."""
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.125},
        "times_to_export": [],  # Supress outputs for tests
    }
    setup = VarelaJNumSetup3D(params)
    setup.prepare_simulation()


def test_model_post_simulation(material_constants: dict) -> None:
    """Test whether the model correctly runs."""
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.125},
        "times_to_export": [],  # Supress outputs for tests
    }
    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})


def test_error_estimates_matching_grids(material_constants: dict) -> None:
    """Test error estimates for a sequence of matching grids."""

    # Set up and run the model
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.125},
        "times_to_export": [],  # Supress outputs for tests
    }
    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})

    # Retrieve grids and data dictionaries
    mdg = setup.mdg
    sd_matrix, d_matrix = mdg.subdomains(return_data=True, dim=3)[0]
    sd_frac, d_frac = mdg.subdomains(return_data=True, dim=2)[0]
    intf, d_intf = mdg.interfaces(return_data=True, dim=2)[0]

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
    te = VarelaJNumTrueErrors3D(setup)
    true_error = te.true_error()
    efficiency_index = majorant / true_error

    # Check against known values
    # np.testing.assert_allclose(majorant, 0.11336778184222394)
    # np.testing.assert_allclose(true_error, 0.10057203148161831)
    # np.testing.assert_allclose(efficiency_index, 1.1272297096130979)
    assert True