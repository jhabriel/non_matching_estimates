"""
Non-matching coupling figure for the 3D/2D VarelaJNum example *with* transfer grids.

Shows ONLY geometry (no estimator colormaps):

Layers (exploded in x):
  - x = 0.5 - Δ : Matrix trace (IBG)            — transparent faces + colored edges
  - x = 0.5     : Interface side grid (Γ1/Γ2)   — transparent faces + colored edges
  - x = 0.5 + Δ : Fracture grid (Ω1)            — transparent faces + colored edges

Transfer grids (as wire overlays):
  - x = 0.5 - Δ/2 : Transfer(IBG → interface side)
  - x = 0.5 + Δ/2 : Transfer(fracture → interface side)

"""

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mdnme
from mdnme.models.varela_jnum_2d.model import manu_incomp_fluid, manu_incomp_solid
from mdnme.models.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.utils.internal_boundary_grid import InternalBoundaryGrid
from mdnme.utils.transfer_grid import TransferGrid

matplotlib.use("Agg")

# ------------- CONFIG -------------
H = 0.15
SIDE = "left"  # "left" (Γ1) or "right" (Γ2)
TRANSLATION = (0, 1, 0)  # used for non-matching run
OUTFILE = "coupling_with_transfers_left.pdf"

# Exploded layout & view
EXPLODE = 0.2  # x-offset between layers
FIGSIZE = (7.0, 3.8)
DPI = 300
PROJ_ORTHO = True  # orthographic projection for diagram look
ELEV = 18.0
AZIM = -60.0
AXES_OFF = True  # hide axes/box/ticks

# Per-layer style (faces are subtle; set to 0.0 for pure wireframes)
EDGE_COLORS = dict(trace="#111111", intf="#1f77b4", frac="#d62728")
EDGE_WIDTHS = dict(trace=0.6, intf=0.6, frac=0.6)
FACE_ALPHA = dict(trace=0.10, intf=0.10, frac=0.10)

# Transfer-grid wire styles
TG_EDGE_COLORS = dict(ibg2msg="#2ca02c", frac2msg="#9467bd")  # green/purple
TG_EDGE_WIDTHS = dict(ibg2msg=0.8, frac2msg=0.8)
TG_FACE_ALPHA = 0.0  # keep transfers as wireframes
# ----------------------------------


def export_transfer_grids(mdg, side: str, out_prefix: str = "tg") -> None:
    """Export the two transfer grids as 2D PDFs via TransferGrid.plot()."""
    # pull grids
    ((sd_mat, _),) = mdg.subdomains(dim=3, return_data=True)
    ((sd_frac, _),) = mdg.subdomains(dim=2, return_data=True)
    ((intf, _),) = mdg.interfaces(dim=2, return_data=True)

    # pick side (Γ1/Γ2) and IBG on that side
    side_enum = _pick_side_enum(intf, side)
    sidegrid = intf.side_grids[side_enum]
    ibg = InternalBoundaryGrid(intf, sd_mat, tol=1e-8)
    ibg_side = ibg.ibg_side_grid(side_enum)
    if ibg_side.num_cells == 0:
        raise RuntimeError("IBG chosen side has zero cells")

    # use the sidegrid orientation for BOTH transfer grids (shared 2D frame)
    R_side = mdnme.RotatedGrid(sidegrid).rotation_matrix

    # build transfers with explicit rotation + distinct names (controls PDF filenames)
    tg_ibg2msg = TransferGrid(
        g_source=ibg_side,
        g_target=sidegrid,
        rotation_matrix=R_side,
        tol=1e-8,
        name=f"{out_prefix}_ibg2{side}",
    )
    tg_frac2msg = TransferGrid(
        g_source=sd_frac,
        g_target=sidegrid,
        rotation_matrix=R_side,
        tol=1e-8,
        name=f"{out_prefix}_frac2{side}",
    )

    # export PDFs (uses self.name+".pdf")
    tg_ibg2msg.plot(base_cmap="viridis", alpha=0.95)
    tg_frac2msg.plot(base_cmap="viridis", alpha=0.95)
    print("Saved:", f"{tg_ibg2msg.name}.pdf", f"{tg_frac2msg.name}.pdf")


def _filter_valid_polys3d(polys, area_eps_abs: float = 1e-14):
    """
    Keep only polygons with >=3 vertices, all finite, and non-zero area
    (area computed in the y–z plane since our layers are x≈const).
    """
    valid = []
    for P in polys:
        P = np.asarray(P)
        if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] < 3:
            continue
        if not np.isfinite(P).all():
            continue
        # y–z shoelace area
        y = P[:, 1]
        z = P[:, 2]
        area = 0.5 * np.abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))
        if area > area_eps_abs:
            valid.append(P)
    return valid


# --------- model helpers ----------
def _material_constants():
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


def _build_setup(h: float, *, non_matching: bool, translation=None):
    common_params = {
        "grid_type": "simplex",
        "material_constants": _material_constants(),
        "meshing_arguments": {"cell_size": h},
        "times_to_export": [],  # keep runtime quiet
    }
    if non_matching:
        assert translation is not None
        params = dict(  # type:ignore
            common_params,
            non_matching=True,
            perturb_fracture=True,
            perturb_mortar=True,
            refine_fracture=False,
            refine_mortar=False,
            translation_vector=tuple(int(v) for v in translation),
        )
    else:
        params = common_params
    return VarelaJNumSetup3D(params)  # type:ignore


def _pick_side_enum(intf: pp.MortarGrid, which: str) -> int:
    """Pick left/right interface side by sign of physical normal's x-component."""
    which = which.lower()
    left_enum = right_enum = None
    for enum, sg in intf.side_grids.items():
        rg = mdnme.RotatedGrid(sg)
        inactive = np.where(~rg.dim_bool)[0][0]
        n_phys = rg.rotation_matrix.T[:, inactive]  # (3,)
        if n_phys[0] < 0:
            left_enum = enum
        else:
            right_enum = enum
    if which == "left":
        assert left_enum is not None
        return left_enum  # type:ignore
    elif which == "right":
        assert right_enum is not None
        return right_enum  # type:ignore
    raise ValueError("SIDE must be 'left' or 'right'")


# ---------- plotting utils --------
def _grid_cell_polys2d(g: pp.Grid):
    """List of cell polygons as (Ni x d) arrays in R^d (d=2 or 3 depending on grid)."""
    cn = g.cell_nodes().tocsc()
    polys = []
    for j in range(g.num_cells):
        idx = cn.indices[cn.indptr[j] : cn.indptr[j + 1]]
        polys.append(g.nodes[:, idx].T.copy())
    return polys


def _add_wire_surface3d(ax, g: pp.Grid, *, edge="#333", lw=0.6, face_alpha=0.10):
    """Add a semi-transparent triangulated surface, edges emphasized (nodes are 3D)."""
    polys = _grid_cell_polys2d(g)  # list of (Ni,3)
    polys = _filter_valid_polys3d(polys)  # <- guard against degeneracy
    if not polys:
        return  # nothing to draw, avoid mplot3d crash

    r, g_, b = mcolors.to_rgb(edge)
    face_col = (r, g_, b, float(face_alpha)) if face_alpha > 0 else "none"
    coll = Poly3DCollection(
        polys,
        facecolors=face_col,
        edgecolor=edge,
        linewidths=float(lw),
        zsort="average",
    )
    ax.add_collection3d(coll)


def _shift_nodes_inplace(g: pp.Grid, dx: float):
    """Shift nodes along x in-place; return original copy to restore later."""
    orig = g.nodes.copy()
    g.nodes[0, :] += dx
    return orig


# --- transfer-grid helpers (robust to different attribute names) ---
def _as_grid_from_transfer(tg) -> pp.Grid:
    """Return the PorePy grid that represents the transfer mesh."""
    # Your class exposes it as `transfer` (a pp.TriangleGrid)
    if hasattr(tg, "transfer") and isinstance(tg.transfer, pp.Grid):
        return tg.transfer
    # Fallbacks for other variants
    for attr in ("grid", "g_transfer", "g", "mesh"):
        g = getattr(tg, attr, None)
        if isinstance(g, pp.Grid):
            return g
    if isinstance(tg, pp.Grid):
        return tg
    raise TypeError("Cannot extract a pp.Grid from TransferGrid")


def _embed_transfer_nodes_to_3d(
    g2d_nodes: np.ndarray, sidegrid: pp.Grid, x_plane: float
) -> np.ndarray:
    """
    Embed 2D nodes (in the sidegrid's intrinsic coordinates) into 3D physical space
    on the plane x = x_plane, using the sidegrid orientation.
    """
    # sidegrid plane geometry via RotatedGrid
    rg = mdnme.RotatedGrid(sidegrid)
    R = rg.rotation_matrix  # x_rot = R @ x_phys
    active = np.where(rg.dim_bool)[0]
    inactive = np.where(~rg.dim_bool)[0][0]

    # map (ξ,η) -> (y,z): yz = T_yz @ [ξ,η] + n_yz * c0
    P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    T_yz = P_yz @ R.T[:, active]  # (2,2)
    rot_cc_full = R @ sidegrid.cell_centers  # (3, N)
    c0 = float(np.mean(rot_cc_full[inactive, :]))  # constant plane offset
    n_yz = (P_yz @ R.T[:, inactive]).reshape(2, 1)  # (2,1)

    yz = T_yz @ g2d_nodes + n_yz * c0  # (2, n_nodes)
    x = np.full((1, g2d_nodes.shape[1]), float(x_plane))
    nodes3d = np.vstack([x, yz])  # (3, n_nodes)
    return nodes3d


def _add_transfer_wire(
    ax, tg, sidegrid: pp.Grid, *, x_plane: float, edge="#333", lw=0.8, face_alpha=0.0
):
    """
    Draw the transfer grid as a wireframe on the plane x = x_plane.

    The transfer mesh lives in tg.transfer (2D frame). We embed its nodes to 3D
    using tg._rot_matrix (or the sidegrid rotation as fallback), then clamp x.
    """
    g = _as_grid_from_transfer(tg)  # pp.TriangleGrid in 2D frame
    nodes2d = g.nodes
    R_used = getattr(tg, "_rot_matrix", None)
    if R_used is None:
        R_used = mdnme.RotatedGrid(sidegrid).rotation_matrix

    rg_tmp = mdnme.RotatedGrid(sidegrid, R_used)
    active = np.where(rg_tmp.dim_bool)[0]
    inactive = np.where(~rg_tmp.dim_bool)[0][0]

    rot_cc_full = R_used @ sidegrid.cell_centers
    c0 = float(np.mean(rot_cc_full[inactive, :]))
    P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    T_yz = P_yz @ R_used.T[:, active]
    n_yz = (P_yz @ R_used.T[:, inactive]).reshape(2, 1)

    if nodes2d.shape[0] == 2:
        yz = T_yz @ nodes2d + n_yz * c0
        nodes3d = np.vstack([np.full((1, nodes2d.shape[1]), float(x_plane)), yz])
    elif nodes2d.shape[0] == 3:
        nodes3d = nodes2d.copy()
        nodes3d[0, :] = float(x_plane)
    else:
        return  # unexpected, but safely do nothing

    # Build polygons in 3D and filter degenerates
    cn = g.cell_nodes().tocsc()
    polys = []
    for j in range(g.num_cells):
        idx = cn.indices[cn.indptr[j] : cn.indptr[j + 1]]
        polys.append(nodes3d[:, idx].T)

    polys = _filter_valid_polys3d(polys)  # <- guard against degeneracy
    if not polys:
        return

    face_col = (0, 0, 0, float(face_alpha)) if face_alpha > 0 else "none"
    coll = Poly3DCollection(
        polys,
        facecolors=face_col,
        edgecolor=edge,
        linewidths=float(lw),
        zsort="average",
    )
    ax.add_collection3d(coll)


# -------------- main --------------
def build_mdg(
    h: float, *, non_matching: bool, translation=None
) -> pp.MixedDimensionalGrid:
    """Build the mdg by running the (quick) time-dependent model once."""
    setup = _build_setup(h, non_matching=non_matching, translation=translation)
    pp.run_time_dependent_model(setup, {})  # constructs the grids
    return setup.mdg


def make_figure_with_transfers(mdg: pp.MixedDimensionalGrid, side: str, outfile: str):
    # Pull grids
    ((sd_mat, _),) = mdg.subdomains(dim=3, return_data=True)
    ((sd_frac, _),) = mdg.subdomains(dim=2, return_data=True)
    ((intf, _),) = mdg.interfaces(dim=2, return_data=True)

    # Choose side grid and IBG on that side
    side_enum = _pick_side_enum(intf, side)
    sidegrid = intf.side_grids[side_enum]  # type:ignore
    ibg = InternalBoundaryGrid(intf, sd_mat, tol=1e-8)
    ibg_side = ibg.ibg_side_grid(side_enum)
    if ibg_side.num_cells == 0:
        raise RuntimeError("IBG chosen side has zero cells")

    # Transfer grids: IBG→side, frac→side
    tg_ibg2msg = TransferGrid(g_source=ibg_side, g_target=sidegrid, tol=1e-8)
    tg_frac2msg = TransferGrid(g_source=sd_frac, g_target=sidegrid, tol=1e-8)

    # Exploded view: shift the main layers (transfer grids will be placed halfway)
    ibg_nodes_orig = _shift_nodes_inplace(ibg_side, -EXPLODE)
    side_nodes_orig = _shift_nodes_inplace(sidegrid, 0.0)
    frac_nodes_orig = _shift_nodes_inplace(sd_frac, +EXPLODE)

    try:
        # 3D canvas
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho" if PROJ_ORTHO else "persp")  # type:ignore
        ax.view_init(elev=ELEV, azim=AZIM)  # type:ignore
        ax.set_box_aspect((1, 1, 1))  # type:ignore
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.set_zlim(0, 1)
        if AXES_OFF:
            ax.set_axis_off()
        else:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            # ax.set_zlabel("z")

        # Draw 3 layers
        _add_wire_surface3d(
            ax,  # type:ignore[attr-defined]
            ibg_side,
            edge=EDGE_COLORS["trace"],
            lw=EDGE_WIDTHS["trace"],
            face_alpha=FACE_ALPHA["trace"],
        )
        _add_wire_surface3d(
            ax,  # type:ignore[attr-defined]
            sidegrid,
            edge=EDGE_COLORS["intf"],
            lw=EDGE_WIDTHS["intf"],
            face_alpha=FACE_ALPHA["intf"],
        )
        _add_wire_surface3d(
            ax,  # type:ignore[attr-defined]
            sd_frac,
            edge=EDGE_COLORS["frac"],
            lw=EDGE_WIDTHS["frac"],
            face_alpha=FACE_ALPHA["frac"],
        )

        # Place transfer grids halfway toward the interface plane
        x_msg = float(np.mean(sidegrid.cell_centers[0]))
        x_tg_ibg = x_msg - 0.5 * EXPLODE
        x_tg_frac = x_msg + 0.5 * EXPLODE

        _add_transfer_wire(
            ax,  # type:ignore[attr-defined]
            tg_ibg2msg,
            sidegrid,
            x_plane=x_tg_ibg,
            edge=TG_EDGE_COLORS["ibg2msg"],
            lw=TG_EDGE_WIDTHS["ibg2msg"],
            face_alpha=TG_FACE_ALPHA,
        )
        _add_transfer_wire(
            ax,  # type:ignore[attr-defined]
            tg_frac2msg,
            sidegrid,
            x_plane=x_tg_frac,
            edge=TG_EDGE_COLORS["frac2msg"],
            lw=TG_EDGE_WIDTHS["frac2msg"],
            face_alpha=TG_FACE_ALPHA,
        )

        plt.tight_layout()
        fig.savefig(outfile, dpi=DPI)
        plt.close(fig)
        print(f"[coupling+transfers] Figure saved to: {outfile}")
    finally:
        # Restore nodes
        ibg_side.nodes[:, :] = ibg_nodes_orig
        sidegrid.nodes[:, :] = side_nodes_orig
        sd_frac.nodes[:, :] = frac_nodes_orig


if __name__ == "__main__":
    print(
        f"[coupling+transfers] Building non-matching mdg"
        f" (h={H}, translation={TRANSLATION}, side={SIDE}) …"
    )
    mdg = build_mdg(H, non_matching=True, translation=TRANSLATION)
    exporter = pp.Exporter(mdg, "single_frac", "figs")
    exporter.write_vtu()
    # after mdg = build_mdg(...)
    export_transfer_grids(mdg, side=SIDE, out_prefix="transfer")
    make_figure_with_transfers(mdg, side=SIDE, outfile=OUTFILE)
    print("[coupling+transfers] Done.")
