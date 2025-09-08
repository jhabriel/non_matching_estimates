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

def set_constant_p1_on_grid(g: pp.Grid, c: float) -> np.ndarray:
    """Return P1 coeffs per cell for constant p(x)=c on grid g."""
    if g.dim == 3:
        # (ax, ay, az, b)
        return np.tile(np.array([0.0, 0.0, 0.0, c]), (g.num_cells, 1))
    elif g.dim == 2:
        # (ax, ay, b)
        return np.tile(np.array([0.0, 0.0, c]), (g.num_cells, 1))
    elif g.dim == 1:
        # (a, b)
        return np.tile(np.array([0.0, c]), (g.num_cells, 1))
    else:  # 0D
        return np.tile(np.array([c]), (g.num_cells, 1))

def set_constant_interface_normal_velocity(intf: pp.MortarGrid, data_intf: dict, lam: float):
    """Write mortar fluxes so that normal velocity is constant λ on each mortar cell."""
    flux = lam * intf.cell_volumes  # λ = flux / |cell|
    data_intf["estimates"]["fv_intf_flux"] = flux.copy()


@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}



#@pytest.mark.parametrize("refine_fracture", [False])
#@pytest.mark.parametrize("refine_mortar",  [False])
def run_constant_test(setup_params, c_high, c_low, lam_mode="zero"):
    # build model (matching or non-matching via params)
    setup = VarelaJNumSetup3D(setup_params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    # ensure canonical frames (IBG/transfer use this)
    assign_canonical_rotations(mdg)

    # grab the single 3D–2D interface
    intf = mdg.interfaces(dim=2)[0]
    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_intf = mdg.interface_data(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low  = mdg.subdomain_data(sd_low)

    # overwrite reconstructed pressures with constants
    data_high["estimates"]["recon_sd_pressure"] = set_constant_p1_on_grid(sd_high, c_high)
    data_low ["estimates"]["recon_sd_pressure"] = set_constant_p1_on_grid(sd_low , c_low )

    # effective k on mortar (scalar or array)
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    if np.isscalar(eff_perm):
        k_mortar = float(eff_perm) * np.ones((intf.num_cells, 1))
    else:
        k_mortar = np.asarray(eff_perm, dtype=float).reshape(-1, 1)

    # set λ on mortar
    if lam_mode == "zero":
        set_constant_interface_normal_velocity(intf, data_intf, lam=0.0)
    elif lam_mode == "cancel":  # make error identically zero by λ = −k Δp
        # Δp = c_low - c_high everywhere (constant)
        cJ = float(c_low - c_high)
        # normal velocity λ per cell: −k * cJ (k can vary by cell)
        lam_per_cell = (-k_mortar.ravel()) * cJ
        data_intf["estimates"]["fv_intf_flux"] = lam_per_cell * intf.cell_volumes
    elif lam_mode == "jump_only":  # λ = 0, Δp = constant ⇒ known non-zero
        set_constant_interface_normal_velocity(intf, data_intf, lam=0.0)
    else:
        raise ValueError("lam_mode ∈ {'zero','cancel','jump_only'}")

    # compute interface diffusive error with both paths
    diff_match = _interface_diffusive_error_2d(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )
    diff_nonmt = _interface_diffusive_error_2d_nonmatching(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )
    return intf, k_mortar.ravel(), diff_match, diff_nonmt

def test_constants_matching_vs_nonmatching(material_constants):
    base = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "times_to_export": [],
        "refine_fracture": False,
        "refine_mortar": False,
    }

    # --- Matching configuration
    params_match = dict(base, non_matching=False)
    # A) zero error: cH=cL, λ=0
    intf, k, dM, dN = run_constant_test(params_match, c_high=1.23, c_low=1.23, lam_mode="zero")
    np.testing.assert_allclose(dM, 0.0, atol=1e-14)
    np.testing.assert_allclose(dN, 0.0, atol=1e-14)

    # B) zero error by cancellation: Δp=cJ, λ=−k·cJ
    intf, k, dM, dN = run_constant_test(params_match, c_high=0.5, c_low=1.1, lam_mode="cancel")
    np.testing.assert_allclose(dM, 0.0, atol=1e-14)
    np.testing.assert_allclose(dN, 0.0, atol=1e-14)

    # C) non-zero known value: λ=0, Δp=cJ
    cH, cL = 0.0, 2.0
    intf, k, dM, dN = run_constant_test(params_match, c_high=cH, c_low=cL, lam_mode="jump_only")
    # expected per-cell: k * (cL - cH)^2 * |cell|
    expected = k * (cL - cH)**2 * intf.cell_volumes
    np.testing.assert_allclose(dM, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dN, expected, rtol=1e-12, atol=1e-14)

    #TODO: Resume from here ... Cannot use matching estimators on non-matching grids

    # --- Non-matching configuration (e.g. refined mortar)
    params_nonmatch = dict(base, non_matching=True, refine_mortar=True)
    # Repeat A/B/C; matching and non-matching estimators should coincide cellwise
    _, _, dM_A, dN_A = run_constant_test(params_nonmatch, c_high=1.0, c_low=1.0, lam_mode="zero")
    np.testing.assert_allclose(dN_A, dM_A, rtol=1e-13, atol=1e-14)

    _, _, dM_B, dN_B = run_constant_test(params_nonmatch, c_high=0.5, c_low=1.25, lam_mode="cancel")
    np.testing.assert_allclose(dN_B, dM_B, rtol=1e-13, atol=1e-14)

    _, _, dM_C, dN_C = run_constant_test(params_nonmatch, c_high=0.0, c_low=3.0, lam_mode="jump_only")
    np.testing.assert_allclose(dN_C, dM_C, rtol=1e-13, atol=1e-14)
