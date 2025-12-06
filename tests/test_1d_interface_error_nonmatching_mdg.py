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

    mdg_coarse = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": 0.2},
        fracture_network=fn,
    )

    mdg_fine = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": 0.1},
        fracture_network=fn,
    )

    data_map = {}
    for sd_coarse, sd_fine in zip(
        mdg_coarse.subdomains(dim=1), mdg_fine.subdomains(dim=1)
    ):
        data_map[sd_coarse] = sd_fine

    mdg_coarse.replace_subdomains_and_interfaces(data_map)

    # Set canonical frames once.
    build_canonical_frames(mdg_coarse)

    return mdg_coarse


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

                xloc = x_loc[:, nodes_k]  # (2, 3)
                z = x_phys[2, nodes_k]  # z-coordinates
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


def random_smooth_p1_on_grid(sd: pp.Grid, rng: np.random.Generator) -> np.ndarray:
    """Construct a random but smoothed P1 field on sd using Oswald-type averaging.

    Returns
    -------
    coeffs : np.ndarray
        Shape (sd.num_cells, sd.dim+1). P1 coefficients in the rotated (canonical)
        coordinates of `sd`, i.e.
          dim=1: p(s) = a*s + b
          dim=2: p(x,y) = a_x*x + a_y*y + c
        on each cell.
    """
    dim = sd.dim
    if dim not in (1, 2):
        raise ValueError("Only 1D and 2D grids are supported here.")

    cn = sd.cell_nodes().tocsc()
    n_cells = sd.num_cells
    n_nodes = sd.num_nodes

    # (1) Start with random cellwise scalars (discontinuous)
    cell_vals = rng.standard_normal(n_cells)

    # (2) Oswald averaging: nodal values = average of incident cell values
    node_sums = np.zeros(n_nodes)
    node_counts = np.zeros(n_nodes, dtype=int)

    for c in range(n_cells):
        i0, i1 = cn.indptr[c], cn.indptr[c + 1]
        nodes_c = cn.indices[i0:i1]
        node_sums[nodes_c] += cell_vals[c]
        node_counts[nodes_c] += 1

    # Avoid division by zero (shouldn’t happen in a sensible grid)
    node_counts = np.maximum(node_counts, 1)
    node_vals = node_sums / node_counts  # continuous nodal field

    # (3) Fit P1 per cell in the *rotated* coordinates
    g_rot = rotate_grid(sd)
    x_loc = g_rot.nodes  # (dim, n_nodes)

    coeffs = np.empty((n_cells, dim + 1))

    for c in range(n_cells):
        i0, i1 = cn.indptr[c], cn.indptr[c + 1]
        nodes_c = cn.indices[i0:i1]

        if dim == 2:
            # Simplex grid: 3 nodes per cell
            assert nodes_c.size == 3, "Expected 2D simplex grid."
            xloc = x_loc[:, nodes_c]  # (2,3)
            vals = node_vals[nodes_c]  # (3,)

            # Solve [x y 1] [a_x, a_y, c]^T = vals
            V = np.vstack((xloc, np.ones(3)))  # (3,3)
            coeffs[c, :] = np.linalg.solve(V.T, vals)

        else:  # dim == 1
            assert nodes_c.size == 2, "Expected 1D simplex grid (segments)."
            n0, n1 = nodes_c
            s0, s1 = x_loc[0, n0], x_loc[0, n1]
            u0, u1 = node_vals[n0], node_vals[n1]

            if abs(s1 - s0) < 1e-14:
                a = 0.0
                b = u0
            else:
                a = (u1 - u0) / (s1 - s0)
                b = u0 - a * s0
            coeffs[c, :] = [a, b]

    return coeffs


def set_constant_k_and_lambda(
    intf: pp.MortarGrid, data_intf: dict, k0: float = 1.0, lambda0: float = 0.0
) -> None:
    """Constant permeability and constant normal velocity on the interface."""
    n = intf.num_cells

    data_intf.setdefault(pp.PARAMETERS, {})
    data_intf[pp.PARAMETERS].setdefault("flow", {})

    # Constant k
    data_intf[pp.PARAMETERS]["flow"]["effective_permeability"] = k0 * np.ones((n, 1))

    # Constant lambda -> mortar flux = lambda * volume
    data_intf.setdefault("estimates", {})
    cell_volumes = intf.cell_volumes
    fv_intf_flux = lambda0 * cell_volumes
    data_intf["estimates"]["fv_intf_flux"] = fv_intf_flux


def set_random_k_and_lambda(
    intf: pp.MortarGrid, data_intf: dict, rng: np.random.Generator
) -> None:
    """Assign random positive permeabilities and random normal velocities."""
    n = intf.num_cells

    # Random positive k: log-normal-ish
    log_k = rng.standard_normal(n)
    k_vals = np.exp(log_k)  # strictly positive
    data_intf.setdefault(pp.PARAMETERS, {})
    data_intf[pp.PARAMETERS].setdefault("flow", {})
    data_intf[pp.PARAMETERS]["flow"]["effective_permeability"] = k_vals.reshape(n, 1)

    # Random normal velocities lambda (can be highly discontinuous)
    lambda_vals = rng.standard_normal(n)  # velocities
    cell_volumes = intf.cell_volumes
    fv_intf_flux = lambda_vals * cell_volumes
    data_intf.setdefault("estimates", {})
    data_intf["estimates"]["fv_intf_flux"] = fv_intf_flux


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field_type", ["constant", "linear", "parabolic"])
def test_nonmatching_1d_error_manufactured_fields(mdg_crossing, field_type):
    """Behaviour of the non-matching 1D estimator for manufactured p(z)."""
    _assign_reconstructed_pressure(mdg_crossing, field_type)
    _set_zero_flux_and_unit_perm(mdg_crossing)

    all_diffs = []

    for intf, data_intf in mdg_crossing.interfaces(return_data=True):
        if intf.dim != 1:
            continue

        sd_high, sd_low = mdg_crossing.interface_to_subdomain_pair(intf)
        data_high = mdg_crossing.subdomain_data(sd_high)
        data_low = mdg_crossing.subdomain_data(sd_low)

        diff = _interface_diffusive_error_1d_nonmatching(
            intf, data_intf, sd_high, data_high, sd_low, data_low
        )
        all_diffs.append(diff)

    all_diffs = np.concatenate(all_diffs) if all_diffs else np.array([])

    if field_type in ("constant", "linear"):
        # P1-exact: the nonmatching machinery should give ~0 cellwise
        np.testing.assert_allclose(all_diffs, 0.0, atol=1.0e-10)
    else:  # "parabolic"
        # SZ / reconstruction not exact for quadratic, so we expect a non-zero residual
        # (beyond pure roundoff). Threshold can be mild.
        assert all_diffs.size > 0
        assert np.any(np.abs(all_diffs) > 1.0e-8), (
            "Parabolic manufactured field unexpectedly gives ~0 diffusive error "
            "with the non-matching estimator."
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_smooth_1d_nonmatching_estimator_stable(mdg_crossing, seed):
    """On non-matching 1D interfaces, the non-matching estimator should run
    and produce finite, non-trivial values for random smoothed P1 fields.
    """
    rng = np.random.default_rng(seed)

    # Make sure canonical frames are set
    build_canonical_frames(mdg_crossing)

    # 1) Assign random smoothed P1 pressures on all 1D and 2D subdomains
    for sd, d in mdg_crossing.subdomains(return_data=True):
        d.setdefault("estimates", {})
        if sd.dim in (1, 2):
            coeffs = random_smooth_p1_on_grid(sd, rng)
            d["estimates"]["recon_sd_pressure"] = coeffs

    # 2) Loop over 1D interfaces and call *only* the non-matching estimator
    for intf, data_intf in mdg_crossing.interfaces(return_data=True):
        if intf.dim != 1:
            continue

        # constant k, zero lambda here; you could also call set_random_k_and_lambda
        set_constant_k_and_lambda(intf, data_intf, 1.0, 0.0)

        sd_high, sd_low = mdg_crossing.interface_to_subdomain_pair(intf)
        data_high = mdg_crossing.subdomain_data(sd_high)
        data_low = mdg_crossing.subdomain_data(sd_low)

        diff_nonmatch = _interface_diffusive_error_1d_nonmatching(
            intf, data_intf, sd_high, data_high, sd_low, data_low
        )

        # Basic sanity checks: shape, finiteness, non-triviality
        assert diff_nonmatch.shape == (intf.num_cells,)
        assert np.all(np.isfinite(diff_nonmatch))
        # With random P1 + non-matching geometry we expect something non-zero
        assert np.any(np.abs(diff_nonmatch) > 1e-12)
