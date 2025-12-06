import numpy as np
import porepy as pp
import pytest

import mdnme
from mdnme.models.varela_jnum_2d.model import manu_incomp_fluid, manu_incomp_solid
from mdnme.models.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.utils.grid_rotation import build_canonical_frames, canonical_frame


@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


@pytest.mark.parametrize("refine_fracture", [True, False])
@pytest.mark.parametrize("refine_mortar", [True, False])
def test_build_canonical_rotations(
    material_constants,
    refine_mortar,
    refine_fracture,
) -> None:
    # With the new Geometry logic, non_matching=True requires at least one of
    # the refine/perturb flags to be True. Here we only use refinement, so we
    # set non_matching to the OR of the refinement flags.
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "non_matching": refine_fracture or refine_mortar,
        "refine_fracture": refine_fracture,
        "refine_mortar": refine_mortar,
        "times_to_export": [],  # Suppress outputs for tests
    }
    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    # Assign canonical rotations before error estimation
    build_canonical_frames(mdg)

    # Grab one 3D subdomain, one 2D fracture, one 2D interface
    sd_high = mdg.subdomains(dim=3)[0]
    sd_low = mdg.subdomains(dim=2)[0]
    intf = mdg.interfaces(dim=2)[0]

    # ---- Checks on highest-dimensional grid ---------------------------------
    R_high, dim_bool_high, dim_high = canonical_frame(sd_high)
    np.testing.assert_array_equal(R_high, np.eye(3))
    np.testing.assert_array_equal(
        dim_bool_high,
        np.array([True, True, True], dtype=bool),
    )
    assert dim_high == 3

    # ---- Checks on fracture surface grid ------------------------------------
    # Canonical frame
    R_low, dim_bool_low, dim_low = canonical_frame(sd_low)
    # Reference rotation from "raw" mapped geometry
    sd_low_rot = mdnme.RotatedGrid(sd_low)

    np.testing.assert_array_equal(R_low, sd_low_rot.rotation_matrix)
    np.testing.assert_array_equal(
        dim_bool_low,
        np.array([True, True, False], dtype=bool),
    )
    assert dim_low == 2

    # ---- Checks on mortar grid ----------------------------------------------
    R_intf, dim_bool_intf, dim_intf = canonical_frame(intf)

    # Inherited from lower-dimensional neighbour
    np.testing.assert_array_equal(R_intf, R_low)
    np.testing.assert_array_equal(dim_bool_intf, dim_bool_low)
    assert dim_intf == 2

    # ---- Coplanarity in the interface frame ---------------------------------
    for intf2 in mdg.interfaces(dim=2):
        R2, _, _ = canonical_frame(intf2)
        z = (R2 @ intf2.cell_centers)[2, :]
        np.testing.assert_allclose(z - z.mean(), 0.0, atol=1e-10)

    # ---- Inheritance invariant ----------------------------------------------
    for intf2 in mdg.interfaces(dim=2):
        _, lo = mdg.interface_to_subdomain_pair(intf2)
        R_intf2, dim_bool_intf2, _ = canonical_frame(intf2)
        R_lo, dim_bool_lo, _ = canonical_frame(lo)

        np.testing.assert_allclose(R_intf2, R_lo)
        np.testing.assert_array_equal(dim_bool_intf2, dim_bool_lo)

    # ---- Idempotency: calling build_canonical_frames twice is stable --------
    # Snapshot canonical frames for all subdomains and interfaces
    subs = list(mdg.subdomains())
    ints = list(mdg.interfaces())
    all_grids = subs + ints

    before = {
        id(g): canonical_frame(g)[:2] for g in all_grids  # (rot_matrix, dim_bool)
    }

    # Rebuild canonical frames
    build_canonical_frames(mdg)

    after = {id(g): canonical_frame(g)[:2] for g in all_grids}

    for gid in before:
        R0, b0 = before[gid]
        R1, b1 = after[gid]

        np.testing.assert_array_equal(b0, b1)
        if R0 is None:
            assert R1 is None
        else:
            np.testing.assert_allclose(R0, R1)
