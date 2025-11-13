import numpy as np
import porepy as pp
import pytest

from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_1d,
    _interface_diffusive_error_1d_nonmatching,
)
from mdnme.utils.grid_rotation import build_canonical_frames, rotate_grid


# --------------------------------------------------------------------------- #
#  Fixtures and helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def mdg_crossing():
    """3D mdg with two intersecting planes, simplex grids."""
    domain = pp.Domain(
        {
            "xmin": 0,
            "ymin": 0,
            "zmin": 0,
            "xmax": 1,
            "ymax": 1,
            "zmax": 1,
        }
    )

    f1 = pp.PlaneFracture(
        np.array(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.25, 0.25, 0.75, 0.75],
                [0.25, 0.75, 0.75, 0.25],
            ]
        )
    )

    f2 = pp.PlaneFracture(
        np.array(
            [
                [0.25, 0.25, 0.75, 0.75],
                [0.5, 0.5, 0.5, 0.5],
                [0.25, 0.75, 0.75, 0.25],
            ]
        )
    )

    fn = pp.create_fracture_network([f1, f2], domain)

    mdg = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": 0.2},
        fracture_network=fn,
    )

    # Set canonical frames once.
    build_canonical_frames(mdg)

    return mdg


def _global_pressure(z, field_type):
    """Global manufactured pressure p(z) for different degrees."""
    if field_type == "constant":
        c0 = 0.3
        return c0 * np.ones_like(z)

    if field_type == "linear":
        alpha = 1.0
        c0 = 0.3
        return alpha * z + c0

    if field_type == "parabolic":
        a2 = 0.7
        b1 = -0.5
        c0 = 0.1
        return a2 * z**2 + b1 * z + c0

    raise ValueError(f"Unknown field_type '{field_type}'")


def _assign_reconstructed_pressure(mdg, field_type):
    """Fill mdg[sd]['estimates']['recon_sd_pressure'] for dim 1 and 2."""
    for sd, d in mdg.subdomains(return_data=True):
        d.setdefault("estimates", {})

        if sd.dim == 0:
            # Nothing to do.
            continue

        # Physical 3D coordinates.
        x_phys = sd.nodes  # (3, n_nodes)

        # Canonical local coordinates (rotated grid).
        g_rot = rotate_grid(sd)
        x_loc = g_rot.nodes  # (sd.dim, n_nodes)

        cn = sd.cell_nodes().tocsc()

        if sd.dim == 2:
            # P1 coeffs [a_x, a_y, c] in local 2D coords.
            c_cell = np.empty((sd.num_cells, 3))

            for k in range(sd.num_cells):
                i0, i1 = cn.indptr[k], cn.indptr[k + 1]
                nodes_k = cn.indices[i0:i1]
                assert nodes_k.size == 3, "Simplex 2D grid expected."

                xloc = x_loc[:, nodes_k]   # (2, 3)
                z = x_phys[2, nodes_k]     # z-coordinates
                vals = _global_pressure(z, field_type)

                # Solve [x y 1] [a_x, a_y, c]^T = vals
                v_mat = np.vstack((xloc, np.ones(3)))  # (3, 3)
                c_cell[k, :] = np.linalg.solve(v_mat.T, vals)

            d["estimates"]["recon_sd_pressure"] = c_cell

        elif sd.dim == 1:
            # P1 coeffs [a_s, b] in local 1D coordinate s = x_loc[0].
            c_cell = np.empty((sd.num_cells, 2))

            s_all = x_loc[0, :]  # 1D local coordinates

            for k in range(sd.num_cells):
                i0, i1 = cn.indptr[k], cn.indptr[k + 1]
                nodes_k = cn.indices[i0:i1]
                assert nodes_k.size == 2, "Simplex 1D grid expected."

                n0, n1 = nodes_k
                s0, s1 = s_all[n0], s_all[n1]
                z0, z1 = x_phys[2, n0], x_phys[2, n1]

                u0 = _global_pressure(z0, field_type)
                u1 = _global_pressure(z1, field_type)

                if abs(s1 - s0) < 1.0e-14:
                    # Degenerate segment: treat as constant.
                    a = 0.0
                    b = u0
                else:
                    a = (u1 - u0) / (s1 - s0)
                    b = u0 - a * s0

                c_cell[k, :] = [a, b]

            d["estimates"]["recon_sd_pressure"] = c_cell

        else:
            # 3D subdomains: not needed for 1D interface test.
            continue


def _set_zero_flux_and_unit_perm(mdg):
    """Set zero mortar flux and unit permeability on all interfaces."""
    for intf, d in mdg.interfaces(return_data=True):
        d.setdefault("estimates", {})
        d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)

        d.setdefault(pp.PARAMETERS, {})
        d[pp.PARAMETERS].setdefault("flow", {})
        d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field_type", ["constant", "linear", "parabolic"])
def test_nonmatching_1d_error_zero_on_matching_grids(mdg_crossing, field_type):
    """On matching grids, the error returned by the non-matching machinery ~ 0"""
    _assign_reconstructed_pressure(mdg_crossing, field_type)
    _set_zero_flux_and_unit_perm(mdg_crossing)

    for intf, data_intf in mdg_crossing.interfaces(return_data=True):
        if intf.dim != 1:
            continue

        sd_high, sd_low = mdg_crossing.interface_to_subdomain_pair(intf)
        data_high = mdg_crossing.subdomain_data(sd_high)
        data_low = mdg_crossing.subdomain_data(sd_low)

        diff = _interface_diffusive_error_1d_nonmatching(
            intf, data_intf, sd_high, data_high, sd_low, data_low
        )

        assert np.allclose(diff, 0.0, atol=1.0e-10)