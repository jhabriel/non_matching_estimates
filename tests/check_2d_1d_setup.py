import numpy as np
import porepy as pp
import mdnme

from mdnme.utils.grid_rotation import assign_canonical_rotations
from mdnme.utils.grid_utils import refine_grid
from mdnme.utils.transfer_grid import TransferLine, TransferGrid
from mdnme.utils.primal_projections import project_p1_1d_sz
from porepy.grids.refinement import GridSequenceFactory

from porepy.fracs.fracture_network_2d import FractureNetwork2d


def build_2d_cross_fractures(
        target_mesh_size: float,
    ) -> tuple[pp.MixedDimensionalGrid, FractureNetwork2d]:

    # Define a rectangular domain in terms of range in the two dimensions
    bounding_box = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
    domain = pp.Domain(bounding_box=bounding_box)

    # Define each individual fracture, collect into a list.
    frac1 = pp.LineFracture(np.array([[0.25, 0.75],
                                      [0.25, 0.75]]))
    frac2 = pp.LineFracture(np.array([[0.25, 0.75],
                                      [0.75, 0.25]]))
    fractures = [frac1, frac2]

    # Define a fracture network in 2d
    fn = pp.create_fracture_network(fractures, domain)

    # Set overall target cell size and target cell size close to the fracture.
    mesh_args: dict[str, float] = {"cell_size": target_mesh_size}

    # Generate a mixed-dimensional grid
    mdg = pp.create_mdg("simplex", mesh_args, fn)

    # Assign rotations
    assign_canonical_rotations(mdg)

    return mdg, fn


def create_non_matching_grid(
        mdg_coarse: pp.MixedDimensionalGrid,
        mdg_fine: pp.MixedDimensionalGrid,
    ) -> pp.MixedDimensionalGrid:

    base_mdg = mdg_coarse.copy()

    # Create mapping
    sd_map = {}
    for sd_base, sd_fine in zip(base_mdg.subdomains(), mdg_fine.subdomains()):

        # Sanity check
        dim_base = sd_base.dim
        dim_fine = sd_fine.dim
        assert dim_base == dim_fine
        dim = dim_base

        if dim == 1:
            sd_map[sd_base] = sd_fine

    # Perform replacement
    base_mdg.replace_subdomains_and_interfaces(sd_map=sd_map)

    # Assign canonical rotations to all subdomain and interfaces
    assign_canonical_rotations(base_mdg)

    return base_mdg


def print_side_health(mdg, tol=1e-12):
    for intf, d in mdg.interfaces(return_data=True, dim=1):
        print("\nInterface (1D) with", intf.num_cells, "mortar cells")
        vols = intf.cell_volumes
        print("  min|max mortar cell length:", vols.min(), vols.max())
        for side in intf.sides:
            mg_side = intf.side_grids[side]
            v = mg_side.cell_volumes
            print("    side:", side, "  n_cells:", mg_side.num_cells,
                  "  min|max side length:", (v.min() if v.size else None, v.max() if v.size else None))


def probe_transferline_on_sides(mdg, tol=1e-10):
    for intf, d in mdg.interfaces(return_data=True, dim=1):
        # IBG for this interface (parent = higher-dim subdomain)
        sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
        from mdnme.utils.internal_boundary_grid import InternalBoundaryLineGrid
        ibg = InternalBoundaryLineGrid(intf, sd_high, tol=tol)

        for P_msg, mg_side in intf.project_to_side_grids():
            side_enum = next(k for k, v in intf.side_grids.items() if v is mg_side)
            ibg_side = ibg.ibg_side_grid(side_enum)

            # share the mortar’s canonical rotation
            R = getattr(intf, "rotation_matrix", None)

            try:
                tl = TransferLine(ibg_side, mg_side, tol=tol, rotation_matrix=R)
                print(f"TL ok: intf-side {side_enum}: n_tr={tl.transfer.num_cells} ",
                      f"span_src=({tl._breaks(ibg_side)[0]:.6g},{tl._breaks(ibg_side)[-1]:.6g}) ",
                      f"span_tgt=({tl._breaks(mg_side)[0]:.6g},{tl._breaks(mg_side)[-1]:.6g})")
            except Exception as e:
                print(f"TL FAIL: intf-side {side_enum}: {e}")


import numpy as np
import porepy as pp
import mdnme

# --------- from earlier: robust scalar wrapper and P1 fitters ---------

def _as_scalar_fn(u):
    def f(x, y):
        try:
            out = u(x, y)
        except TypeError:
            out = u(np.array([x, y]))
        out = np.asarray(out)
        if out.shape == ():
            return out.item()
        if out.size == 1:
            return float(out.ravel()[0])
        raise ValueError(f"u(x,y) must be scalar; got shape {out.shape}")
    return f

def _tri_cell_node_indices(grid: pp.Grid) -> np.ndarray:
    cn = grid.cell_nodes().tocsc()
    if cn.nnz % 3 != 0:
        raise ValueError("Grid is not purely triangular (nnz % 3 != 0).")
    return cn.indices.reshape((3, grid.num_cells), order="F").T

def _seg_cell_node_indices(grid: pp.Grid) -> np.ndarray:
    fn = grid.cell_nodes().tocsc()
    if fn.nnz % 2 != 0:
        raise ValueError("1D grid is not purely segments (nnz % 2 != 0).")
    return fn.indices.reshape((2, grid.num_cells), order="F").T

def fit_p1_per_cell(grid: pp.Grid, u) -> np.ndarray:
    """
    2D triangles: returns (nc × 3) [a,b,c], u_h(x,y)=a x + b y + c
    1D segments : returns (nc × 2) [a,b],   u_h(s)  = a s + b  (s in canonical frame)
    """
    u = _as_scalar_fn(u)
    rot = mdnme.RotatedGrid(grid)
    Xg  = rot.nodes[:max(2, grid.dim), :]

    if grid.dim == 2:
        tri = _tri_cell_node_indices(grid)
        C   = np.empty((int(grid.num_cells), 3), dtype=float)
        for k, (n0, n1, n2) in enumerate(tri):
            x0, y0 = float(Xg[0, n0]), float(Xg[1, n0])
            x1, y1 = float(Xg[0, n1]), float(Xg[1, n1])
            x2, y2 = float(Xg[0, n2]), float(Xg[1, n2])
            V   = np.array([[x0, y0, 1.0],
                            [x1, y1, 1.0],
                            [x2, y2, 1.0]], dtype=float)
            rhs = np.array([u(x0, y0), u(x1, y1), u(x2, y2)], dtype=float)
            try:
                C[k, :] = np.linalg.solve(V, rhs)
            except np.linalg.LinAlgError:
                C[k, :] = np.linalg.lstsq(V, rhs, rcond=None)[0]
        return C

    if grid.dim == 1:
        # s uses the same canonical rotation for consistency
        s_coords = mdnme.RotatedGrid(grid, rot.rotation_matrix).nodes[0, :]
        seg = _seg_cell_node_indices(grid)
        C   = np.empty((int(grid.num_cells), 2), dtype=float)
        for k, (n0, n1) in enumerate(seg):
            s0, s1 = float(s_coords[n0]), float(s_coords[n1])
            x0, y0 = float(Xg[0, n0]),    float(Xg[1, n0])
            x1, y1 = float(Xg[0, n1]),    float(Xg[1, n1])
            v0 = u(x0, y0); v1 = u(x1, y1)
            h = s1 - s0
            if np.isclose(h, 0.0):
                a, b = 0.0, 0.5 * (v0 + v1)
            else:
                a = (v1 - v0) / h
                b = v0 - a * s0
            C[k, 0] = a; C[k, 1] = b
        return C

    raise NotImplementedError("Only 1D/2D supported.")

# --------- seeding utilities ---------

def _ensure_estimates(d):
    est = d.get("estimates")
    if est is None:
        est = {}
        d["estimates"] = est
    return est

def _side_name(side_enum) -> str:
    # Helpful, readable key names per side
    try:
        return str(side_enum).split(".")[-1].lower()
    except Exception:
        return str(side_enum)

# --------- main seeding function ---------

def seed_synthetic_estimates(
    mdg: pp.MixedDimensionalGrid,
    pressure_fn,
    *,
    k_bulk: float = 1.0,         # matrix/fracture tangential permeability (in-plane/along-line)
    k_normal: float = 1.0,       # only used if you later compute normal fluxes at interfaces
) -> None:
    """
    Populate each subdomain and each 1D interface side with synthetic pressure
    and simple Darcy flux proxies.

    Writes:
      Subdomain (dim=2):
        data["estimates"]["pressure_p1"]  -> (nc,3) [a,b,c]
        data["estimates"]["flux_tangent"] -> (nc,2) q = -k_bulk * [a,b]
      Subdomain (dim=1):
        data["estimates"]["pressure_p1"]  -> (nc,2) [a,b] for u(s)=a s + b
        data["estimates"]["flux_tangent"] -> (nc,)   q_s = -k_bulk * a

      Interface (dim=1): for each side S
        data_intf["estimates"][f"trace_pressure_p1_{S}"] -> (nc_side,2) on that side grid

    Notes:
      - Uses the canonical rotated frames (consistent with your TransferGrid/Line).
      - Traces are seeded by *direct fitting on the mortar side grid* using the
        same pressure function, to keep things simple and robust for debugging.
    """
    p = _as_scalar_fn(pressure_fn)

    # ---- subdomains ----
    for sd, d in mdg.subdomains(return_data=True):
        est = _ensure_estimates(d)
        if sd.dim == 2:
            C = fit_p1_per_cell(sd, p)           # (nc,3)
            est["pressure_p1"] = C
            # Darcy flux proxy: q = -K ∇p, here K = k_bulk I
            q = -k_bulk * C[:, :2]               # (nc,2), columns [a,b]
            est["flux_tangent"] = q
        elif sd.dim == 1:
            C = fit_p1_per_cell(sd, p)           # (nc,2) in s
            est["pressure_p1"] = C
            est["flux_tangent"] = (-k_bulk * C[:, 0]).copy()   # (nc,)

    # ---- interfaces (only 1D are meaningful in 2D problems) ----
    for intf, d in mdg.interfaces(return_data=True, dim=1):
        estI = _ensure_estimates(d)

        # Each side has its own 1D grid; we seed the trace by fitting on that grid.
        for side in intf.sides:
            side_grid = intf.side_grids[side]
            C_side = fit_p1_per_cell(side_grid, p)         # (nc_side,2)
            key = f"trace_pressure_p1_{_side_name(side)}"
            estI[key] = C_side

        # Optional: store mortar cell lengths (handy for quick sanity checks)
        estI.setdefault("mortar_cell_length", intf.cell_volumes.copy())

# Build non-matching grid (your helpers)
mdg_coarse, _ = build_2d_cross_fractures(0.10)
mdg_fine,   _ = build_2d_cross_fractures(0.05)
mdg_nm        = create_non_matching_grid(mdg_coarse, mdg_fine)

# Choose a smooth pressure
def p2(x, y):              # quadratic is nice (exact under your degree-4 quad rule)
    return x*x + 0.3*x*y + 0.5*y*y + 0.2*x - 0.1*y + 0.05

assign_canonical_rotations(mdg_nm)  # already in your build() but safe here too
seed_synthetic_estimates(mdg_nm, p2, k_tangential=1.0, k_normal=1.0)

# Run your existing estimator
from mdnme.estimates.error_estimation import compute_diffusive_error
compute_diffusive_error(mdg_nm, non_matching_nested=False)

# Inspect: everything should be ~0 (machine epsilon / quad error)
for sd, d in mdg_nm.subdomains(return_data=True):
    de = d["estimates"]["diffusive_error"]
    print(f"SD dim={sd.dim}  max(diffusive)={de.max():.3e}")

for intf, d in mdg_nm.interfaces(return_data=True):
    de = d["estimates"]["diffusive_error"]
    print(f"INTF dim={intf.dim} max(diffusive)={de.max():.3e}")
