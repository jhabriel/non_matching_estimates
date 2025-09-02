import mdnme
import porepy as pp
import numpy as np
import pytest

from porepy.grids.mortar_grid import MortarSides
from porepy.grids.refinement import GridSequenceFactory
from mdnme.utils.grid_rotation import assign_canonical_rotations

from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import (
    manu_incomp_fluid,
    manu_incomp_solid,
    )


@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


@pytest.mark.parametrize('refine_fracture', [True, False])
@pytest.mark.parametrize('refine_mortar', [True, False])
def test_assign_canonical_rotations(
        material_constants,
        refine_mortar,
        refine_fracture,
) -> None:
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "non_matching": True,
        "refine_fracture": refine_fracture,
        "refine_mortar": refine_mortar,
        "times_to_export": [],  # Supress outputs for tests
    }
    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    # Assign canonical rotations before error estimation
    assign_canonical_rotations(mdg)

    # Check whether we have the rotation matrices
    sd_high = mdg.subdomains(dim=3)[0]
    sd_low = mdg.subdomains(dim=2)[0]
    intf = mdg.interfaces(dim=2)[0]

    # Check on highest-dimensional grid
    np.testing.assert_array_equal(sd_high.rot_matrix, np.eye(3))
    np.testing.assert_array_equal(
        np.array(sd_high.dim_bool, dtype=bool),
        np.array([True, True, True], dtype=bool),
    )

    # Check on fracture surface grid
    sd_low_rot = mdnme.RotatedGrid(sd_low)
    np.testing.assert_array_equal(sd_low.rot_matrix, sd_low_rot.rotation_matrix)
    np.testing.assert_array_equal(
        np.array(sd_low.dim_bool, dtype=bool),
        np.array([True, True, False], dtype=bool),
    )

    # Check on mortar grid
    np.testing.assert_array_equal(intf.rot_matrix, sd_low_rot.rotation_matrix)
    np.testing.assert_array_equal(
        np.array(intf.dim_bool, dtype=bool),
        np.array([True, True, False], dtype=bool),
    )

    # Coplanarity in the interface frame
    for intf in mdg.interfaces(dim=2):
        R = intf.rot_matrix
        z = (R @ intf.cell_centers)[2, :]
        np.testing.assert_allclose(z - z.mean(), 0.0, atol=1e-10)

    # Inheritance invariant
    for intf in mdg.interfaces(dim=2):
        _, lo = mdg.interface_to_subdomain_pair(intf)
        np.testing.assert_allclose(intf.rot_matrix, lo.rot_matrix)
        np.testing.assert_array_equal(intf.dim_bool, lo.dim_bool)

    # Idempotency
    # call twice; frames shouldn’t change
    before = [(id(g), getattr(g, "rot_matrix", None)) for g in mdg.subdomains()]
    assign_canonical_rotations(mdg)
    after = [(id(g), getattr(g, "rot_matrix", None)) for g in mdg.subdomains()]
    for (_, R0), (_, R1) in zip(before, after):
        if R0 is None:
            assert R1 is None
        else:
            np.testing.assert_allclose(R0, R1)
