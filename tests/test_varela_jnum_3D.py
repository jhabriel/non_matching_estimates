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
from porepy.grids.refinement import GridSequenceFactory

from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import (
    manu_incomp_fluid,
    manu_incomp_solid,
    )
from mdnme.examples.varela_jnum_3d.true_errors import VarelaJNumTrueErrors3D


@pytest.fixture(scope="module")
def grid_sequence() -> list[pp.MixedDimensionalGrid]:
    """Create an mdg sequence of refined grids."""
    domain = pp.Domain(
        {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1})
    frac = pp.PlaneFracture(np.array([
        [0.50, 0.50, 0.50, 0.50],
        [0.25, 0.75, 0.75, 0.25],
        [0.25, 0.25, 0.75, 0.75],
    ]))
    fn = pp.create_fracture_network([frac], domain)

    mesh_args = {  # coarsish base; factory will refine
        "mesh_size_bound": 0.4,
        "mesh_size_frac": 0.4,
        "mesh_size_min": 0.01,
    }
    params = {"mode": "nested", "num_refinements": 2, "mesh_param": mesh_args}
    factory = GridSequenceFactory(fn, params)
    mdgs = list(factory)
    # pick the 2D fracture subdomain from each MDG
    # levels = [mdg.subdomains()[1] for mdg in mdgs]
    return mdgs  # coarse -> ... -> finest


@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


@pytest.fixture(scope="module")
def desired_errors_matching_grids() -> list[dict]:
    """Hardcoded values of desired errors."""
    desired_errors_025 = {
        'majorant': 0.2553744582135114,
        'true_error': 0.21754078829532522,
        'eff_idx': 1.173915292918883,
    }
    desired_errors_0125 = {
        'majorant': 0.16947899720717424,
        'true_error': 0.15259723183109286,
        'eff_idx': 1.1106295649895372,
    }
    desired_errors_00625 = {
        'majorant': 0.11887565856452474,
        'true_error': 0.11314512602221603,
        'eff_idx': 1.0506476305588588,
    }
    return [desired_errors_025, desired_errors_0125, desired_errors_00625]


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


@pytest.mark.parametrize('cell_size', [0.25, 0.125, 0.0625])
def test_error_estimates_matching_grids(
        material_constants: dict,
        desired_errors_matching_grids: list[dict],
        cell_size: float
        ) -> None:
    """Test error estimates for a sequence of matching grids."""

    # Set up and run the model
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": cell_size},
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
    eff_idx = majorant / true_error

    # Check against desired values
    if cell_size == 0.25:
        check_idx = 0
    elif cell_size == 0.125:
        check_idx = 1
    else:
        check_idx = 2

    # Retrieved desired values
    majorant_desired = desired_errors_matching_grids[check_idx]['majorant']
    true_error_desired = desired_errors_matching_grids[check_idx]['true_error']
    eff_idx_desired = desired_errors_matching_grids[check_idx]['eff_idx']

    # Assert
    assert np.isclose(majorant, majorant_desired, 1e-5, 1e-4)
    assert np.isclose(true_error, true_error_desired, 1e-5, 1e-4)
    assert np.isclose(eff_idx, eff_idx_desired, 1e-5, 1e-4)


def test_non_matching_assembly_3d(grid_sequence):
    """Checks whether a non-matching grid is correctly assembled."""

    mdg_coarse = grid_sequence[0]
    mdg_fine = grid_sequence[1]

    sd_matrix_coarse = mdg_coarse.subdomains(dim=3)[0]
    sd_frac_coarse = mdg_coarse.subdomains(dim=2)[0]
    sd_frac_fine = mdg_fine.subdomains(dim=2)[0]
    intf_coarse = mdg_coarse.interfaces(dim=2)[0]

    mdg_coarse.replace_subdomains_and_interfaces(sd_map={sd_frac_coarse: sd_frac_fine})

    # Checks whether the updated mdg has the correct number of cells
    assert mdg_coarse.subdomains(dim=3)[0].num_cells == sd_matrix_coarse.num_cells
    assert mdg_coarse.subdomains(dim=2)[0].num_cells == sd_frac_fine.num_cells
    assert mdg_coarse.interfaces(dim=2)[0].num_cells == intf_coarse.num_cells
