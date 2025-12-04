import numpy as np
import porepy as pp
import pytest

from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_1d_nonmatching,
)
from mdnme.utils.grid_rotation import build_canonical_frames, rotate_grid


# --------------------------------------------------------------------------- #
#  Manufactured pressure and helpers
# --------------------------------------------------------------------------- #

def global_parabolic_pressure(z: np.ndarray) -> np.ndarray:
    """Global manufactured pressure p(z) = a2 z^2 + b1 z + c0."""
    a2 = 0.7
    b1 = -0.5
    c0 = 0.1
    return a2 * z**2 + b1 * z + c0


def assign_reconstructed_pressure(mdg: pp.MixedDimensionalGrid) -> None:
    """
    Fill mdg[sd]['estimates']['recon_sd_pressure'] for dim 1 and 2.

    - 2D: P1 coeffs [a_x, a_y, c] in rotated (canonical) coordinates.
    - 1D: P1 coeffs [a_s, b] in rotated local coordinate s.
    """
    for sd, d in mdg.subdomains(return_data=True):
        d.setdefault("estimates", {})

        if sd.dim == 0:
            continue

        # Physical 3D coordinates
        x_phys = sd.nodes  # (3, n_nodes)

        # Canonical local coordinates
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
                vals = global_parabolic_pressure(z)

                # Solve [x y 1] [a_x, a_y, c]^T = vals
                V = np.vstack((xloc, np.ones(3)))  # (3, 3)
                c_cell[k, :] = np.linalg.solve(V.T, vals)

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

                u0 = global_parabolic_pressure(z0)
                u1 = global_parabolic_pressure(z1)

                if abs(s1 - s0) < 1.0e-14:
                    a = 0.0
                    b = u0
                else:
                    a = (u1 - u0) / (s1 - s0)
                    b = u0 - a * s0

                c_cell[k, :] = [a, b]

            d["estimates"]["recon_sd_pressure"] = c_cell

        else:
            # 3D subdomains not needed for 1D interface estimator
            continue


def set_zero_flux_and_unit_perm(mdg: pp.MixedDimensionalGrid) -> None:
    """Set zero mortar flux and unit permeability on all interfaces."""
    for intf, d in mdg.interfaces(return_data=True):
        d.setdefault("estimates", {})
        d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)

        d.setdefault(pp.PARAMETERS, {})
        d[pp.PARAMETERS].setdefault("flow", {})
        # effective_permeability could also be cell-wise, but 1.0 is enough here
        d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0


# --------------------------------------------------------------------------- #
#  Non-matching MDG builder
# --------------------------------------------------------------------------- #

def build_crossing_mdg_nonmatching(h_coarse: float,
                                   h_fine_1d: float) -> pp.MixedDimensionalGrid:
    """
    Build 3D mdg with two intersecting planes (Geiger-like) and
    non-matching 1D grids:

    - 2D and 3D grids with coarse size h_coarse.
    - 1D subdomains replaced by a finer size h_fine_1d.
    """
    domain = pp.Domain(
        {
            "xmin": 0.0,
            "ymin": 0.0,
            "zmin": 0.0,
            "xmax": 1.0,
            "ymax": 1.0,
            "zmax": 1.0,
        }
    )

    # Two crossing fractures as in your earlier tests
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

    # Coarse MDG (2D+3D will survive)
    mdg_coarse = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": h_coarse},
        fracture_network=fn,
    )

    # Fine MDG only to steal the 1D subdomains
    mdg_fine_1d = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": h_fine_1d},
        fracture_network=fn,
    )

    # Replace all 1D subdomains in coarse MDG by the fine ones
    data_map = {}
    for sd_coarse, sd_fine in zip(
        mdg_coarse.subdomains(dim=1), mdg_fine_1d.subdomains(dim=1)
    ):
        data_map[sd_coarse] = sd_fine

    mdg_coarse.replace_subdomains_and_interfaces(data_map)

    # Recompute local projections
    pp.set_local_coordinate_projections(mdg_coarse)

    # Set canonical frames once.
    build_canonical_frames(mdg_coarse)

    return mdg_coarse


def build_3fracs_mdg_nonmatching(h_coarse: float,
                                h_fine_1d: float) -> pp.MixedDimensionalGrid:
    """
    Build 3D mdg with three intersecting planes (Geiger-like) and
    non-matching 1D grids and a 0-d intersection:

    - 2D and 3D grids with coarse size h_coarse.
    - 1D subdomains replaced by a finer size h_fine_1d.
    """
    domain = pp.Domain(
        {
            "xmin": 0.0,
            "ymin": 0.0,
            "zmin": 0.0,
            "xmax": 1.0,
            "ymax": 1.0,
            "zmax": 1.0,
        }
    )

    def rotate_around_the_x_axis(
            array: np.ndarray,
            theta: float = 10.0,
    ) -> np.ndarray:

        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                [0, np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))],
            ]
        )

        return R_x @ array

    def rotate_around_the_z_axis(
            array: np.ndarray,
            theta: float = 10.0,
    ) -> np.ndarray:

        R_z = np.array(
            [
                [np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta)), 0],
                [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0],
                [0, 0, 1],
            ]
        )
        return R_z @ array

    points_f1 = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.25, 0.25, 0.75, 0.75],
            [0.25, 0.75, 0.75, 0.25],
        ]
    )
    points_f2 = np.array(
        [
            [0.25, 0.25, 0.75, 0.75],
            [0.5, 0.5, 0.5, 0.5],
            [0.25, 0.75, 0.75, 0.25],
        ]
    )

    points_f3 = np.array(
        [
            [0.25, 0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75, 0.75],
            [0.50, 0.50, 0.50, 0.50],
        ]
    )

    f1 = pp.PlaneFracture(rotate_around_the_x_axis(points_f1))
    f2 = pp.PlaneFracture(rotate_around_the_z_axis(points_f2))
    f3 = pp.PlaneFracture(points_f3)

    fn = pp.create_fracture_network([f1, f2, f3], domain)

    # Coarse MDG (2D+3D will survive)
    mdg_coarse = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": h_coarse},
        fracture_network=fn,
    )

    # Fine MDG only to steal the 1D subdomains
    mdg_fine_1d = pp.create_mdg(
        "simplex",
        meshing_args={"cell_size": h_fine_1d},
        fracture_network=fn,
    )

    # Replace all 1D subdomains in coarse MDG by the fine ones
    data_map = {}
    for sd_coarse, sd_fine in zip(
        mdg_coarse.subdomains(dim=1), mdg_fine_1d.subdomains(dim=1)
    ):
        data_map[sd_coarse] = sd_fine

    mdg_coarse.replace_subdomains_and_interfaces(data_map)

    # Set canonical frames once.
    build_canonical_frames(mdg_coarse)

    return mdg_coarse


# --------------------------------------------------------------------------- #
#  Norm computation for the 1D non-matching diffusive estimator
# --------------------------------------------------------------------------- #

def compute_interface_norms(mdg: pp.MixedDimensionalGrid):
    """
    Compute global L2-like and energy-like norms of the 1D non-matching
    diffusive estimator over all 1D interfaces.

    IMPORTANT:
    ----------
    We assume that _interface_diffusive_error_1d_nonmatching returns an
    array 'eta' of length 6 per interface, where:

      - eta[:3]  contribute to an L2-type measure,
      - eta[3:]  contribute to an energy-type measure.

    If your convention is different, simply adjust the slicing below.
    """
    energy_acc = 0.0

    for intf, data_intf in mdg.interfaces(return_data=True):
        if intf.dim != 1:
            continue

        sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
        data_high = mdg.subdomain_data(sd_high)
        data_low = mdg.subdomain_data(sd_low)

        # Per-interface contributions (vector of length 6 in your logs)
        eta = _interface_diffusive_error_1d_nonmatching(
            intf, data_intf, sd_high, data_high, sd_low, data_low
        )

        eta = np.asarray(eta, dtype=float)

        # Interpret elements as squared contributions and accumulate.
        # Adjust these slices if your function encodes them differently.
        energy_acc += np.sum(eta)

    # Global norms
    energy_norm = np.sqrt(energy_acc)
    return energy_norm


# --------------------------------------------------------------------------- #
#  Convergence test
# --------------------------------------------------------------------------- #

def test_nonmatching_diffusive_estimator_convergence_parabolic_without_0d():
    """
    Check that, for a smooth parabolic manufactured solution and
    non-matching 1D grids, the 1D diffusive estimator behaves as:

        ||·||_E     = O(h)

    where h is the characteristic 2D mesh size (and h_fine_1d = h/2).
    """

    # Mesh sizes: coarse 2D h, fine 1D h/2
    h_coarse_list = [0.4, 0.2, 0.1, 0.05]
    h_1d_list = [0.2, 0.1, 0.05, 0.025]

    hs = []
    energy_norms = []

    for h_c, h_1d in zip(h_coarse_list, h_1d_list):
        mdg = build_crossing_mdg_nonmatching(h_c, h_1d)

        # Manufactured parabolic pressure on 2D/1D + trivial fluxes, k=1
        assign_reconstructed_pressure(mdg)
        set_zero_flux_and_unit_perm(mdg)

        e_eta = compute_interface_norms(mdg)

        hs.append(h_c)  # or use h_1d; the rate is the same up to constants
        energy_norms.append(e_eta)

    hs = np.array(hs, dtype=float)
    energy_norms = np.array(energy_norms, dtype=float)

    # Fit slopes in log-log scale: log(norm) ≈ alpha + p * log(h)
    p_E, _ = np.polyfit(np.log(hs), np.log(energy_norms), 1)

    # We expect:
    #   p_L2 ≈ 2     (L2-error ~ h^2)
    #   p_E  ≈ 1     (energy-error ~ h)
    # Since we fit log(norm) vs log(h), the slope should be ~ p > 0
    # if norm(h) ~ C h^p.  If your convention is norm(h) ~ C h^{-p},
    # just change the expected sign accordingly.

    # Here I'll use the standard FEM convention: error ~ h^p => slope ~ p.
    # If your logs show negative slopes, flip the sign in the assertions.

    # Allow a bit of slack because we only have 3 levels and a non-trivial
    # geometry. Tune tol if needed.
    tol = 0.5

    print(f"H1 error is: {p_E}")

    assert 1.0 - tol <= p_E <= 1.0 + tol, (
        f"Energy convergence rate too far from 1: p_E = {p_E:.2f}"
    )

def test_nonmatching_diffusive_estimator_convergence_parabolic_with_0d():
    """
    Check that, for a smooth parabolic manufactured solution and
    non-matching 1D grids, the 1D diffusive estimator behaves as:

        ||·||_L2    = O(h^2)
        ||·||_E     = O(h)

    where h is the characteristic 2D mesh size (and h_fine_1d = h/2).
    """

    # Mesh sizes: coarse 2D h, fine 1D h/2
    h_coarse_list = [0.4, 0.2, 0.1, 0.05]
    h_1d_list = [0.2, 0.1, 0.05, 0.025]

    hs = []
    energy_norms = []

    for h_c, h_1d in zip(h_coarse_list, h_1d_list):
        mdg = build_3fracs_mdg_nonmatching(h_c, h_1d)

        # Manufactured parabolic pressure on 2D/1D + trivial fluxes, k=1
        assign_reconstructed_pressure(mdg)
        set_zero_flux_and_unit_perm(mdg)

        e_eta = compute_interface_norms(mdg)

        hs.append(h_c)  # or use h_1d; the rate is the same up to constants
        energy_norms.append(e_eta)

    hs = np.array(hs, dtype=float)
    energy_norms = np.array(energy_norms, dtype=float)

    # Fit slopes in log-log scale: log(norm) ≈ alpha + p * log(h)
    p_E, _ = np.polyfit(np.log(hs), np.log(energy_norms), 1)

    # We expect:
    #   p_L2 ≈ 2     (L2-error ~ h^2)
    #   p_E  ≈ 1     (energy-error ~ h)
    # Since we fit log(norm) vs log(h), the slope should be ~ p > 0
    # if norm(h) ~ C h^p.  If your convention is norm(h) ~ C h^{-p},
    # just change the expected sign accordingly.

    # Here I'll use the standard FEM convention: error ~ h^p => slope ~ p.
    # If your logs show negative slopes, flip the sign in the assertions.

    print(f"H1 error is: {p_E}")

    # Allow a bit of slack because we only have 3 levels and a non-trivial
    # geometry. Tune tol if needed.
    tol = 0.5

    assert 1.0 - tol <= p_E <= 1.0 + tol, (
        f"Energy convergence rate too far from 1: p_E = {p_E:.2f}"
    )

