import numpy as np
import porepy as pp
import pytest

import mdnme
from mdnme.utils.internal_boundary_grid import InternalBoundaryLineGrid
from mdnme.estimates.diffusive_error import _interface_diffusive_error_1d_nonmatching

# --- helpers to make a 3D network with 2D fractures intersecting in 1D ---
def make_two_crossing_fractures_unit_cube():
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1})
    # Two planes that intersect in a straight line roughly through the cube
    F1 = pp.PlaneFracture(np.array([
        [0.2, 0.8, 0.8, 0.2],
        [0.2, 0.2, 0.8, 0.8],
        [0.5, 0.5, 0.5, 0.5],
    ]))
    F2 = pp.PlaneFracture(np.array([
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.8, 0.8, 0.2],
        [0.2, 0.2, 0.8, 0.8],
    ]))
    fn = pp.create_fracture_network([F1, F2], domain)
    mdg = pp.create_mdg(
        "simplex",
        {
            "cell_size_boundary": 0.6,
            "cell_size_fracture": 0.15,
            "cell_size_min": 0.03,
        },
        fn,
    )
    return mdg

@pytest.mark.parametrize("kval", [1.0, 3.0])
def test_1d_interface_error_zero_when_no_jump(kval):
    mdg = make_two_crossing_fractures_unit_cube()

    # pick one 2D–1D interface (there should be at least one)
    intf = mdg.interfaces(dim=1)[0]
    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low  = mdg.subdomain_data(sd_low)
    data_intf = mdg.interface_data(intf)

    # try to ensure canonical rotations exist; if not, skip nicely
    if not hasattr(intf, "rot_matrix") or intf.rot_matrix is None:
        pytest.skip("Canonical rotations not available on this MortarGrid in the test environment.")

    # fabricate reconstructed pressures consistent across the interface:
    # take p(s) = a s + b in the *canonical* 1D coordinate; choose a,b
    a, b = -0.9, 0.3

    # HIGH (2D) pressure per cell: p(x,y) = a * s(x,y) + b; approximate with local P1
    # We do this by sampling the 2D cell vertices in the interface frame and solving a small LSQ
    gh_rot = mdnme.RotatedGrid(sd_high)  # just to access node coords
    Xh = gh_rot.nodes[:2, :]
    cn_h = sd_high.cell_nodes().tocsc()
    cells_h = cn_h.indices.reshape((sd_high.dim + 1, sd_high.num_cells), order="F").T
    C_high = np.empty((sd_high.num_cells, 3))
    # define "s" as the first canonical axis in the interface frame projected to high grid
    R = intf.rot_matrix; dim_bool = np.array(intf.dim_bool, dtype=bool)
    nodes2d_high_in_mortar = (R @ sd_high.nodes)[dim_bool, :]

    for K, verts in enumerate(cells_h):
        xy = nodes2d_high_in_mortar[:, verts]  # project to interface plane
        svals = xy[0, :]                       # take the first axis as 1D coordinate
        uvals = a * svals + b
        V = np.vstack((Xh[:, verts], np.ones(3)))
        C_high[K, :] = np.linalg.solve(V.T, uvals)

    # LOW (1D) pressure per cell: exact p(s) = a s + b
    xl = sd_low.nodes[0, :]
    C_low = np.empty((sd_low.num_cells, 2))
    for k in range(sd_low.num_cells):
        s0, s1 = xl[k], xl[k+1]
        u0, u1 = a*s0 + b, a*s1 + b
        ak = (u1 - u0)/(s1 - s0)
        bk = u0 - ak*s0
        C_low[k, :] = [ak, bk]

    # stash into data dicts where the estimator looks for them
    data_high.setdefault("estimates", {})["recon_sd_pressure"] = C_high
    data_low.setdefault("estimates", {})["recon_sd_pressure"] = C_low

    # interface fluxes = 0 -> λ = 0 => with zero jump, error must be ~0
    data_intf.setdefault("estimates", {})["fv_intf_flux"] = np.zeros(intf.num_cells)
    data_intf.setdefault(pp.PARAMETERS, {}).setdefault("flow", {})["effective_permeability"] = kval

    # run the 1D nonmatching estimator (IBG➜TL➜SZ inside)
    err2 = _interface_diffusive_error_1d_nonmatching(
        intf, data_intf, sd_high, data_high, sd_low, data_low, tol=1e-8
    )
    # all zeros up to roundoff
    assert err2.shape == (intf.num_cells,)
    assert np.all(err2 >= -1e-12)
    assert np.max(np.abs(err2)) < 1e-10
