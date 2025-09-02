import numpy as np
import porepy as pp
import mdnme
import pytest

from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import (
    manu_incomp_fluid,
    manu_incomp_solid,
    )
from mdnme.utils.internal_boundary_grid import InternalBoundaryGrid
from mdnme.utils.transfer_grid import TransferGrid
from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_2d,               # matching path (your existing one)
    _interface_diffusive_error_2d_nonmatching,   # new path
)
from mdnme.utils.grid_rotation import assign_canonical_rotations
from mdnme.utils.primal_projections import restrict_to_transfer, scott_zhang_quasi_interpolant

# --- helper: assert SZ(TG,·) is identity for matching grids on P1 ---
def assert_sz_identity_on_p1(grid: pp.Grid, tol=1e-12):
    TG = TransferGrid(grid, grid)  # identical
    # fabricate random-but-fixed P1 coeffs (a,b,c) per cell
    rng = np.random.default_rng(0)
    p1 = rng.standard_normal(size=(grid.num_cells, 3))
    # restrict→SZ back
    p_on_tg = restrict_to_transfer(TG, p1)
    p_rec   = scott_zhang_quasi_interpolant(TG, p_on_tg)
    np.testing.assert_allclose(p_rec, p1, rtol=1e-12, atol=tol)

@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}



#@pytest.mark.parametrize("refine_fracture", [False])
#@pytest.mark.parametrize("refine_mortar",  [False])
def test_nonmatching_estimator_equals_matching_when_grids_match(material_constants):
    # ----- build a setup where mortar == fracture trace mesh (matching) -----
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "non_matching": False,         # IMPORTANT: matching configuration
        "refine_fracture": False,
        "refine_mortar": False,
        "times_to_export": [],
    }
    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    mdnme.estimate_errors(mdg)


    # pick the (single) 3D–2D interface
    intf = mdg.interfaces(dim=2)[0]
    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_intf = mdg.interface_data(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low  = mdg.subdomain_data(sd_low)

    # sanity: SZ projector is identity on P1 for identical grids
    assert_sz_identity_on_p1(sd_low)
    for _, mg_side in intf.project_to_side_grids():
        assert_sz_identity_on_p1(mg_side)

    # ----- compute estimators -----
    diff_matching = _interface_diffusive_error_2d(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )
    diff_nonmatch = _interface_diffusive_error_2d_nonmatching(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )

    # ----- compare (cellwise) with tight tolerance -----
    np.testing.assert_allclose(diff_nonmatch, diff_matching, rtol=1e-10, atol=1e-12)

    # also check global sums
    np.testing.assert_allclose(diff_nonmatch.sum(), diff_matching.sum(), rtol=1e-12, atol=1e-14)
