"""
Grid-mismatch (GM) error indicators for non-matching mixed-dimensional grids.

At k=0 (FV/RT0), only η_GM,DF,⊥ and the lower-dim contribution to η_GM,R are
nontrivial; η_GM,DF,∥ and the higher-dim η_GM,R contribution vanish identically
(paper Remark~rem:gm_lowest_order).

η²_GM,DF,⊥,K is computed exactly by constructing the common refinement T_{ǰ,ĵ}
of the two transfer grids T_{Γ_j,Ω_ǰ} (fracture→mortar) and T_{Γ_j,∂_j Ω_ĵ}
(IBG→mortar). On each cell of T_{ǰ,ĵ} both defects δP_ǰ and δP_ĵ are P1
polynomials, so κ|δP_ǰ − δP_ĵ|² is P2 and integrates exactly by a degree-4
quadpy scheme.

Reference:
    Varela, J., et al. A posteriori error estimates for non-matching
    mixed-dimensional elliptic equations. arXiv preprint (2025).

"""

from __future__ import annotations

import numpy as np
import porepy as pp
import quadpy
import scipy.sparse as sps

import mdnme
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from mdnme.estimates.diffusive_error import _get_high_pressure_trace
from mdnme.utils.grid_utils import ear_clip_triangulate
from mdnme.utils.internal_boundary_grid import (
    InternalBoundaryGrid,
    InternalBoundaryLineGrid,
)
from mdnme.utils.primal_projections import (
    prolong_to_transfer,
    prolong_to_transfer_1d,
    scott_zhang_quasi_interpolant,
    scott_zhang_quasi_interpolant_1d,
)
from mdnme.utils.transfer_grid import TransferGrid, TransferLine


def compute_gm_diffusive_parallel(mdg: pp.MixedDimensionalGrid) -> None:
    """Compute the squared GM diffusive parallel indicator η²_GM,DF,∥ per subdomain cell.

    At k=0 (FV/RT0), this is identically zero for all interfaces. Stored in
    ``data_sd["estimates"]["gm_diffusive_error_parallel"]`` on every subdomain.
    """
    for sd, data_sd in mdg.subdomains(return_data=True):
        data_sd["estimates"]["gm_diffusive_error_parallel"] = np.zeros(sd.num_cells)

    for intf, data_intf in mdg.interfaces(return_data=True):
        sd_high, _ = mdg.interface_to_subdomain_pair(intf)
        data_high = mdg.subdomain_data(sd_high)
        if intf.dim == 2:
            val = _gm_diffusive_parallel_2d(intf, data_intf, sd_high, data_high)
            data_high["estimates"]["gm_diffusive_error_parallel"] += val


def _gm_diffusive_parallel_2d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    tol: float = 1e-5,
) -> np.ndarray:
    """Returns zero array — η²_GM,DF,∥ vanishes identically at k=0 (FV/RT0)."""
    return np.zeros(sd_high.num_cells)


def compute_gm_residual(mdg: pp.MixedDimensionalGrid) -> None:
    """Compute the squared GM residual indicator η²_GM,R per subdomain cell.

    At k=0, bulk cells are zero; fracture cells are nontrivial.
    Results stored in ``data_sd["estimates"]["gm_residual_error"]`` on every subdomain.
    """
    for sd, data_sd in mdg.subdomains(return_data=True):
        data_sd["estimates"]["gm_residual_error"] = np.zeros(sd.num_cells)

    for intf, data_intf in mdg.interfaces(return_data=True):
        _, sd_low = mdg.interface_to_subdomain_pair(intf)
        data_low = mdg.subdomain_data(sd_low)
        if intf.dim == 2:
            val = _gm_residual_fracture_2d(intf, data_intf, sd_low, data_low)
            data_low["estimates"]["gm_residual_error"] += val
        elif intf.dim == 1:
            val = _gm_residual_fracture_1d(intf, data_intf, sd_low, data_low)
            data_low["estimates"]["gm_residual_error"] += val


def _gm_residual_fracture_2d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_low: pp.Grid,
    data_low: dict,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute η²_GM,R per fracture cell for a 2D non-matching interface at k=0."""
    rot_matrix, _, _ = mdnme.canonical_frame(intf)

    fv_intf_flux = data_intf["estimates"]["fv_intf_flux"]
    mortar_flux_density = fv_intf_flux / intf.cell_volumes

    perm = data_low[pp.PARAMETERS]["flow"]["second_order_tensor"].values
    k_low = perm[0, 0, :]
    h_low = sd_low.cell_diameters()

    out = np.zeros(sd_low.num_cells)

    for P_msg, mg_side in intf.project_to_side_grids():
        tg = TransferGrid(
            g_source=sd_low, g_target=mg_side, rotation_matrix=rot_matrix, tol=tol
        )
        t2s = tg.transfer_to_source.tocsr()
        t2t = tg.transfer_to_target.tocsr()

        td_to_frac = np.asarray(t2s.argmax(axis=1)).ravel()
        td_to_side = np.asarray(t2t.argmax(axis=1)).ravel()

        side_to_global = np.asarray(P_msg.tocsr().argmax(axis=1)).ravel()
        td_to_global = side_to_global[td_to_side]

        tr_areas = tg.transfer.cell_volumes

        for K in np.unique(td_to_frac):
            mask = td_to_frac == K
            areas_K = tr_areas[mask]
            dens_K = mortar_flux_density[td_to_global[mask]]
            total_area = areas_K.sum()
            if total_area < tol:
                continue

            mean_K = np.dot(areas_K, dens_K) / total_area
            delta_sq = np.dot(areas_K, (dens_K - mean_K) ** 2)

            c_K = np.sqrt(max(float(k_low[K]), 0.0))
            if c_K < tol:
                continue

            prefactor = (h_low[K] / (np.pi * c_K)) ** 2
            out[K] += prefactor * delta_sq

    return out


def _gm_residual_fracture_1d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_low: pp.Grid,
    data_low: dict,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute η²_GM,R per 1D intersection cell for a 1D non-matching interface at k=0."""
    rot_matrix, dim_bool, _ = mdnme.canonical_frame(intf)

    fv_intf_flux = data_intf["estimates"]["fv_intf_flux"]
    mortar_flux_density = fv_intf_flux / intf.cell_volumes

    perm = data_low[pp.PARAMETERS]["flow"]["second_order_tensor"].values
    k_low = perm[0, 0, :]
    h_low = sd_low.cell_diameters()

    out = np.zeros(sd_low.num_cells)

    for P_msg, mg_side in intf.project_to_side_grids():
        tl = TransferLine(
            sd_low,
            mg_side,
            rotation_matrix=rot_matrix,
            dim_bool=dim_bool,
        )
        t2s = tl.transfer_to_source.tocsr()
        t2t = tl.transfer_to_target.tocsr()

        td_to_low = np.asarray(t2s.argmax(axis=1)).ravel()
        td_to_side = np.asarray(t2t.argmax(axis=1)).ravel()

        side_to_global = np.asarray(P_msg.tocsr().argmax(axis=1)).ravel()
        td_to_global = side_to_global[td_to_side]

        tr_lengths = tl.transfer.cell_volumes

        for K in np.unique(td_to_low):
            mask = td_to_low == K
            lengths_K = tr_lengths[mask]
            dens_K = mortar_flux_density[td_to_global[mask]]
            total_length = lengths_K.sum()
            if total_length < tol:
                continue

            mean_K = np.dot(lengths_K, dens_K) / total_length
            delta_sq = np.dot(lengths_K, (dens_K - mean_K) ** 2)

            c_K = np.sqrt(max(float(k_low[K]), 0.0))
            if c_K < tol:
                continue

            prefactor = (h_low[K] / (np.pi * c_K)) ** 2
            out[K] += prefactor * delta_sq

    return out


def compute_gm_diffusive_perp(mdg: pp.MixedDimensionalGrid) -> None:
    """Compute the squared GM diffusive interface indicator η²_GM,DF,⊥ per mortar cell.

    Results stored in ``data_intf["estimates"]["gm_diffusive_error_perp"]``.
    """
    for intf, data_intf in mdg.interfaces(return_data=True):
        sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
        data_high = mdg.subdomain_data(sd_high)
        data_low = mdg.subdomain_data(sd_low)

        if intf.dim == 2:
            val = _gm_diffusive_perp_2d(
                intf, data_intf, sd_high, data_high, sd_low, data_low
            )
        elif intf.dim == 1:
            val = _gm_diffusive_perp_1d(
                intf, data_intf, sd_high, data_high, sd_low, data_low
            )
        else:
            val = np.zeros(intf.num_cells)

        data_intf["estimates"]["gm_diffusive_error_perp"] = val


def _gm_diffusive_perp_1d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute η²_GM,DF,⊥ per mortar cell on a 1D non-matching interface."""
    rot_matrix, dim_bool, _ = mdnme.canonical_frame(intf)

    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    k_mortar = (
        float(eff_perm) * np.ones(intf.num_cells)
        if np.isscalar(eff_perm)
        else np.asarray(eff_perm, dtype=float).ravel()
    )

    p_low_1d = data_low["estimates"]["recon_sd_pressure"]

    frac_edges = sps.find(intf.primary_to_mortar_avg())[1]
    p_trace_high = _get_high_pressure_trace(
        sd_low, sd_high, data_high, frac_edges, rot_matrix, dim_bool
    )
    face2pos = {int(f): i for i, f in enumerate(frac_edges)}

    ibg = InternalBoundaryLineGrid(intf, sd_high, tol=tol)
    out_global = np.zeros(intf.num_cells)

    _xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])

    for P_msg, mg_side in intf.project_to_side_grids():
        side_enum = next(k for k, v in intf.side_grids.items() if v is mg_side)
        ibg_side = ibg.ibg_side_grid(side_enum)
        parent_edges = ibg.parent_edge_of_cell(side_enum)

        if ibg_side.num_cells == 0:
            continue

        idx = np.fromiter(
            (face2pos[int(e)] for e in parent_edges),
            dtype=int,
            count=parent_edges.size,
        )
        tr_hi_on_ibg = p_trace_high[idx, :]

        tl_ibg = TransferLine(
            ibg_side,
            mg_side,
            rotation_matrix=rot_matrix,
            dim_bool=dim_bool,
            rotate_source=False,
            rotate_target=True,
        )
        tl_low = TransferLine(
            sd_low,
            mg_side,
            rotation_matrix=rot_matrix,
            dim_bool=dim_bool,
            rotate_source=True,
            rotate_target=True,
        )

        C_ibg_tr = prolong_to_transfer_1d(tl_ibg, tr_hi_on_ibg)
        C_ibg_mg = scott_zhang_quasi_interpolant_1d(tl_ibg, C_ibg_tr)

        C_low_tr = prolong_to_transfer_1d(tl_low, p_low_1d)
        C_low_mg = scott_zhang_quasi_interpolant_1d(tl_low, C_low_tr)

        k_side = (P_msg @ k_mortar.reshape(-1, 1)).ravel()
        t2t_ibg = np.asarray(
            tl_ibg.transfer_to_target.tocsr().argmax(axis=1)
        ).ravel()

        pts_ibg = tl_ibg.transfer.nodes[0, :]
        pts_low = tl_low.transfer.nodes[0, :]
        all_pts = np.unique(np.concatenate([pts_ibg, pts_low]))
        keep = np.concatenate([[True], np.diff(all_pts) > tol])
        all_pts = all_pts[keep]

        exact_side = np.zeros(mg_side.num_cells)

        for i in range(len(all_pts) - 1):
            s_a, s_b = float(all_pts[i]), float(all_pts[i + 1])
            h_seg = s_b - s_a
            if h_seg <= tol:
                continue
            s_mid = 0.5 * (s_a + s_b)

            t_ibg = int(
                np.clip(np.searchsorted(pts_ibg, s_mid, side="right") - 1,
                        0, tl_ibg.transfer.num_cells - 1)
            )
            t_low = int(
                np.clip(np.searchsorted(pts_low, s_mid, side="right") - 1,
                        0, tl_low.transfer.num_cells - 1)
            )
            K = int(t2t_ibg[t_ibg])

            coeff = (
                C_ibg_tr[t_ibg] - C_ibg_mg[K] - C_low_tr[t_low] + C_low_mg[K]
            )

            s_pts = s_mid + 0.5 * h_seg * _xi
            vals = coeff[0] * s_pts + coeff[1]
            integral = k_side[K] * 0.5 * h_seg * float(np.dot(vals, vals))

            exact_side[K] += integral

        out_global += (P_msg.T @ exact_side).ravel()

    return out_global


def _intersect_transfer_grids_direct(
    tg_fg: TransferGrid,
    tg_ibg: TransferGrid,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute double-transfer data via direct Shapely cell intersections."""
    nodes_fg = tg_fg.transfer.nodes[:2, :]
    nodes_ibg = tg_ibg.transfer.nodes[:2, :]
    cn_fg = tg_fg.transfer.cell_nodes().tocsc()
    cn_ibg = tg_ibg.transfer.cell_nodes().tocsc()
    n_fg = tg_fg.transfer.num_cells
    n_ibg = tg_ibg.transfer.num_cells
    cn_arr_fg = cn_fg.indices.reshape((3, n_fg), order="F")
    cn_arr_ibg = cn_ibg.indices.reshape((3, n_ibg), order="F")

    fg_polys = [Polygon(nodes_fg[:, cn_arr_fg[:, i]].T) for i in range(n_fg)]
    ibg_polys = [Polygon(nodes_ibg[:, cn_arr_ibg[:, j]].T) for j in range(n_ibg)]

    td_to_fg_list: list[int] = []
    td_to_ibg_list: list[int] = []
    elements_list: list[np.ndarray] = []

    tree_ibg = STRtree(ibg_polys)
    for i, poly_fg in enumerate(fg_polys):
        for j in tree_ibg.query(poly_fg):
            poly_ibg = ibg_polys[j]
            inter = poly_fg.intersection(poly_ibg)
            if inter.area <= tol:
                continue
            if hasattr(inter, "exterior"):
                polys = [inter]
            else:
                polys = [g for g in inter.geoms if hasattr(g, "exterior") and g.area > tol]
            for poly in polys:
                raw = list(poly.exterior.coords)[:-1]
                coords: list = []
                for p in raw:
                    if not coords or coords[-1] != p:
                        coords.append(p)
                tris_raw = (
                    [coords] if len(coords) == 3 else ear_clip_triangulate(coords, tol=tol)
                )
                for tri_coords in tris_raw:
                    pts = np.array(tri_coords[:3], dtype=float)
                    area = 0.5 * abs(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
                    if area <= tol:
                        continue
                    td_to_fg_list.append(i)
                    td_to_ibg_list.append(j)
                    elements_list.append(pts)

    td_to_fg = np.array(td_to_fg_list, dtype=int)
    td_to_ibg = np.array(td_to_ibg_list, dtype=int)
    elements_td = np.stack(elements_list, axis=1)
    return td_to_fg, td_to_ibg, elements_td


def _gm_diffusive_perp_2d(
    intf: pp.MortarGrid,
    data_intf: dict,
    sd_high: pp.Grid,
    data_high: dict,
    sd_low: pp.Grid,
    data_low: dict,
    tol: float = 1e-5,
) -> np.ndarray:
    """Compute η²_GM,DF,⊥ per mortar cell on a 2D non-matching interface."""
    rot_matrix, dim_bool, _ = mdnme.canonical_frame(intf)
    method = quadpy.t2.get_good_scheme(4)

    eff_perm = data_intf[pp.PARAMETERS]["flow"]["effective_permeability"]
    k_mortar = (
        float(eff_perm) * np.ones(intf.num_cells)
        if np.isscalar(eff_perm)
        else np.asarray(eff_perm, dtype=float).ravel()
    )

    p_low_frac = data_low["estimates"]["recon_sd_pressure"]
    frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
    p_trace_high = _get_high_pressure_trace(
        sd_low, sd_high, data_high, frac_faces, rot_matrix, dim_bool
    )
    face2pos = {int(f): i for i, f in enumerate(frac_faces)}

    ibg = InternalBoundaryGrid(intf, sd_high, tol=tol)
    out_global = np.zeros(intf.num_cells)

    for P_msg, mg_side in intf.project_to_side_grids():
        side_enum = next(k for k, v in intf.side_grids.items() if v is mg_side)
        ibg_side = ibg.ibg_side_grid(side_enum)
        parent_faces = ibg.parent_face_of_cell(side_enum)

        if ibg_side.num_cells == 0:
            tr_hi_on_ibg = np.zeros((0, 3))
        else:
            idx = np.fromiter(
                (face2pos[int(f)] for f in parent_faces),
                dtype=int,
                count=parent_faces.size,
            )
            tr_hi_on_ibg = p_trace_high[idx, :]

        tg_ibg_msg = TransferGrid(
            g_source=ibg_side, g_target=mg_side, rotation_matrix=rot_matrix, tol=tol
        )
        tg_fg_msg = TransferGrid(
            g_source=sd_low, g_target=mg_side, rotation_matrix=rot_matrix, tol=tol
        )

        tracep_on_tg = prolong_to_transfer(tg_ibg_msg, tr_hi_on_ibg)
        tracep_on_msg = scott_zhang_quasi_interpolant(tg_ibg_msg, tracep_on_tg)

        fracp_on_tg = prolong_to_transfer(tg_fg_msg, p_low_frac)
        fracp_on_msg = scott_zhang_quasi_interpolant(tg_fg_msg, fracp_on_tg)

        k_side = (P_msg @ k_mortar.reshape(-1, 1)).ravel()

        td_to_fg, td_to_ibg, elements_td = _intersect_transfer_grids_direct(
            tg_fg_msg, tg_ibg_msg, tol=tol
        )
        t2tgt_fg = tg_fg_msg.transfer_to_target.tocsr()
        td_to_K = np.asarray(t2tgt_fg[td_to_fg].argmax(axis=1)).ravel()

        diff_coeff = (
            fracp_on_tg[td_to_fg]
            - tracep_on_tg[td_to_ibg]
            + tracep_on_msg[td_to_K]
            - fracp_on_msg[td_to_K]
        )

        k_td = k_side[td_to_K]

        def integrand(x):
            val = mdnme.utils.evaluate_p1(diff_coeff, x)
            return k_td[:, np.newaxis] * val**2

        td_integrals = method.integrate(integrand, elements_td)

        exact_side = np.zeros(mg_side.num_cells)
        np.add.at(exact_side, td_to_K, td_integrals)

        out_global += (P_msg.T @ exact_side).ravel()

    return out_global
