import numpy as np

import mdnme
import porepy as pp
import pytest

from mdnme.models.varela_jnum_3d import VarelaJNumSetup3D
from mdnme.models.varela_jnum_2d.model import manu_incomp_fluid, manu_incomp_solid
from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_2d_nonmatching,
    _interface_diffusive_error_2d,
)
from mdnme.estimates.helpers import is_nonmatching
from typing import Literal


# Helpers
def set_constant_p1_on_grid(g: pp.Grid, c: float) -> np.ndarray:
    """Return per-cell P1 coefficients for the constant field u ≡ c."""
    nc = g.num_cells
    if g.dim == 2:
        return np.column_stack([np.zeros(nc), np.zeros(nc), np.full(nc, c)])
    elif g.dim == 3:
        return np.column_stack([np.zeros(nc), np.zeros(nc),
                                np.zeros(nc), np.full(nc, c)])
    elif g.dim == 1:
        return np.column_stack([np.zeros(nc), np.full(nc, c)])
    else:  # 0d
        return np.full((nc, 1), c)


def set_constant_interface_normal_velocity(
        intf: pp.MortarGrid,
        data_intf: dict,
        lam: float,
    ) -> None:
    """Set λ constant per mortar cell; store integrated flux (λ * |cell|)."""
    lam_cell = np.full(intf.num_cells, lam, dtype=float)
    data_intf["estimates"]["fv_intf_flux"] = lam_cell * intf.cell_volumes


def run_constant_test(
    setup_params: dict,
    c_high: float,
    c_low: float,
    lam_mode: str = "zero",
):
    """Build model, impose constants, compute k and both estimators (dM may be None)."""
    setup = VarelaJNumSetup3D(setup_params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg

    intf = mdg.interfaces(dim=2)[0]
    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_intf = mdg.interface_data(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low  = mdg.subdomain_data(sd_low)

    # overwrite reconstructed pressures with constants
    data_high["estimates"]["recon_sd_pressure"] = set_constant_p1_on_grid(sd_high, c_high)
    data_low ["estimates"]["recon_sd_pressure"] = set_constant_p1_on_grid(sd_low , c_low )

    # effective k on mortar (as vector)
    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    k = (float(eff_perm) * np.ones(intf.num_cells)) if np.isscalar(eff_perm) \
        else np.asarray(eff_perm, dtype=float).ravel()

    # set λ on mortar
    if lam_mode == "zero":
        set_constant_interface_normal_velocity(intf, data_intf, lam=0.0)
    elif lam_mode == "cancel":
        cJ = float(c_low - c_high)
        set_constant_interface_normal_velocity(intf, data_intf, lam=-(k * cJ))
    elif lam_mode == "jump_only":
        set_constant_interface_normal_velocity(intf, data_intf, lam=0.0)
    else:
        raise ValueError("lam_mode ∈ {'zero','cancel','jump_only'}")

    # compute estimators
    nonmatching = is_nonmatching(intf)
    dM = None
    if not nonmatching:
        dM = _interface_diffusive_error_2d(intf, data_intf, sd_high, data_high, sd_low, data_low)
    dN = _interface_diffusive_error_2d_nonmatching(intf, data_intf, sd_high, data_high, sd_low, data_low)

    return intf, k, dM, dN


@pytest.fixture(scope="module")
def material_constants() -> dict:
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


def _base_params(material_constants, *, non_matching, refine_mortar=False, refine_fracture=False):
    return {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "times_to_export": [],
        "non_matching": non_matching,
        "refine_fracture": refine_fracture,
        "refine_mortar": refine_mortar,
    }


@pytest.mark.parametrize(
    "non_matching,refine_mortar,refine_fracture,lam_mode,c_high,c_low",
    [
        # matching cases (non_matching=True just to retrieve the results)
        (True, False, False, "zero"     , 1.23, 1.23),
        (True, False, False, "cancel"   , 0.50, 1.10),
        (True, False, False, "jump_only", 0.00, 2.00),

        # non-matching cases (e.g. refined mortar)
        (True , True , False, "zero"     , 1.00, 1.00),
        (True , True , False, "cancel"   , 0.50, 1.25),
        (True , True , False, "jump_only", 0.00, 3.00),

        # non-matching cases (e.g. refined fracture)
        (True, False, True, "zero", 1.00, 1.00),
        (True, False, True, "cancel", 0.50, 1.25),
        (True, False, True, "jump_only", 0.00, 3.00),

        # non-matching cases (e.g. refined mortar and fracture)
        (True, True, True, "zero", 1.00, 1.00),
        (True, True, True, "cancel", 0.50, 1.25),
        (True, True, True, "jump_only", 0.00, 3.00),
    ]
)
def test_constants_matching_vs_nonmatching(
    material_constants: dict,
    non_matching: bool,
    refine_mortar: bool,
    refine_fracture: bool,
    lam_mode: Literal['zero', 'cancel', 'jump_only'],
    c_high: float,
    c_low: float
):
    params = _base_params(
        material_constants,
        non_matching=non_matching,
        refine_mortar=refine_mortar,
        refine_fracture=refine_fracture
    )

    intf, k, dM, dN = run_constant_test(
        params, c_high=c_high, c_low=c_low, lam_mode=lam_mode
    )

    # analytic expected value when λ=0:  k * (Δp)^2 * |cell|
    if lam_mode == "jump_only":
        expected = k * (c_low - c_high)**2 * intf.cell_volumes
    elif lam_mode == "cancel":
        expected = np.zeros_like(dN)
    else:  # "zero" with c_high==c_low
        expected = np.zeros_like(dN)

    if not non_matching:
        # matching path exists; they must be equal and match the analytic value
        np.testing.assert_allclose(dN, dM, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(dM, expected, rtol=1e-12, atol=1e-14)
    else:
        # no matching estimator here; just check against analytic value
        np.testing.assert_allclose(dN, expected, rtol=1e-12, atol=1e-14)

def test_interface_diffusive_matching_equals_nonmatching_on_matching_grid(material_constants):
    # Matching config
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "times_to_export": [],
        "non_matching": False,
        "refine_fracture": False,
        "refine_mortar": False,
    }

    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg
    mdnme.estimate_errors(mdg)

    # Pick the (single) 3D–2D interface
    intf = mdg.interfaces(dim=2)[0]
    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_intf = mdg.interface_data(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low  = mdg.subdomain_data(sd_low)

    # Compute both estimators using the *actual reconstructed fields*
    dM = _interface_diffusive_error_2d(
        intf,
        data_intf,
        sd_high,
        data_high,
        sd_low,
        data_low
    )
    dN = _interface_diffusive_error_2d_nonmatching(
        intf,
        data_intf,
        sd_high,
        data_high,
        sd_low,
        data_low
    )

    # Cellwise equality (tight), plus global sums
    np.testing.assert_allclose(dN, dM, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dN.sum(), dM.sum(), rtol=1e-14, atol=1e-16)
