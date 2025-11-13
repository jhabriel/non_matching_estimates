"""
Module containing functionality to project P1 potentials from source to target grids.

The two main projection steps are:

    (1) Restriction from the source grid onto the transfer grid.
    (2) Projection from transfer grid onto the target grid using the Scott–Zhang
        quasi‐interpolator.

"""
import mdnme
import numpy as np
import porepy as pp

from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree

from mdnme.utils.transfer_grid import TransferGrid, TransferLine


def restrict_to_transfer(tg: TransferGrid, C_src: np.ndarray) -> np.ndarray:
    """
    From source cell-wise P1 (C_src shape = (n_src_cells,3))
    build transfer cell-wise P1 (C_tr shape = (n_tr_cells,3)) by
    sampling + local fitting.
    """
    g_tr   = tg.transfer
    t2s    = tg.transfer_to_source.tocsr()   # (n_tr_cells × n_src_cells)
    cn_tr  = g_tr.cell_nodes().tocsc()
    tr_cells = cn_tr.indices.reshape((3, g_tr.num_cells), order="F").T

    # Coordinates of transfer nodes (in the rotated plane)
    Xn = g_tr.nodes[:2, :]  # (2, n_tr_nodes)

    C_tr = np.empty((g_tr.num_cells, 3))
    for j, verts in enumerate(tr_cells):
        # 1) parent source cell
        parents = t2s[j,:].nonzero()[1]
        if len(parents) != 1:
            raise RuntimeError(f"Transfer cell {j} has {len(parents)} parents")
        K = parents[0]

        # 2) sample at the 3 vertices
        xy = Xn[:, verts]           # shape (2,3)
        c0, c1, c2 = C_src[K]       # source coeffs
        u = c0*xy[0, :] + c1*xy[1, :] + c2  # length-3

        # 3) solve small Vandermonde
        V = np.vstack((xy, np.ones(3)))    # shape (3,3)
        C_tr[j, :] = np.linalg.solve(V.T, u)  # gives [c0_tr,c1_tr,c2_tr]

    return C_tr


def scott_zhang_quasi_interpolant(tg: TransferGrid, u_tr: np.ndarray) -> np.ndarray:
    """
    Robust Scott–Zhang using coarse-cell quadrature + transfer-mesh point
    evaluations. Exact on P1 and constant-preserving.

    Parameters
    ----------
    tg : TransferGrid
        Transfer grid object (source=transfer, target=grid to reconstruct on).
    u_tr : (n_transfer_cells, 3) ndarray
        Cell-wise P1 coefficients [a, b, c] on the transfer mesh.

    Returns
    -------
    p1_tgt : (n_target_cells, 3) ndarray
        Cell-wise P1 coefficients on the target grid.
    """
    g_tr  = tg.transfer
    g_tgt = tg.g_target

    # --- target (coarse) cell data ---
    cn_tgt   = g_tgt.cell_nodes().tocsc()
    tri_tgt  = cn_tgt.indices.reshape((3, g_tgt.num_cells), order="F").T
    tgt_rot  = tg._get_rotated_grid(g_tgt)
    X_tgt    = tgt_rot.nodes[:2, :]

    coarse_data = []
    for verts in tri_tgt:
        p0, p1, p2 = X_tgt[:, verts[0]], X_tgt[:, verts[1]], X_tgt[:, verts[2]]
        A = np.column_stack((p1 - p0, p2 - p0))
        area = abs(np.linalg.det(A)) * 0.5
        # Inverse of the P1 mass matrix on a triangle: (1/A) * [[9,-3,-3],...]
        M_inv = (1.0 / area) * np.array([[ 9., -3., -3.],
                                         [-3.,  9., -3.],
                                         [-3., -3.,  9.]])
        coarse_data.append((verts, p0, A, M_inv, area))

    # --- transfer mesh barycentric + STRtree ---
    cn_tr   = g_tr.cell_nodes().tocsc()
    tri_tr  = cn_tr.indices.reshape((3, g_tr.num_cells), order="F").T
    tr_rot  = tg._get_rotated_grid(g_tr)
    X_tr    = tr_rot.nodes[:2, :]

    tr_polys = [Polygon(X_tr[:, verts].T) for verts in tri_tr]
    # robust mapping: use id(geom) as key (shapely may return new objects)
    polyid_to_j = {id(geom): j for j, geom in enumerate(tr_polys)}
    tree        = STRtree(tr_polys)

    # prepared geoms, indexed by j
    prepared = [prep(p) for p in tr_polys]

    # barycentric inverses per transfer triangle
    V_inv_tr = []
    for verts in tri_tr:
        V = np.vstack((X_tr[:, verts], np.ones(3)))  # 3x3
        V_inv_tr.append(np.linalg.inv(V))

    # --- SZ nodal values on target ---
    n_tgt_nodes = g_tgt.num_nodes
    u_tgt_nodes = np.empty(n_tgt_nodes)
    n2c = g_tgt.cell_nodes().tocsr()
    # 3-pt quadrature exact for linears
    quadrature = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
    eps = 1e-12

    for i in range(n_tgt_nodes):
        adj = n2c[i].nonzero()[1]
        if adj.size == 0:
            raise RuntimeError(f"Target node {i} has no adjacent cell")

        # standard SZ: pick one adjacent coarse cell as the patch element
        k = int(adj[0])
        verts_k, p0, A, M_inv, area = coarse_data[k]

        # assemble RHS b = ∫_K u φ_i  (via exact 3-pt rule)
        b = np.zeros(3)
        for r, s in quadrature:
            xq = p0 + A @ np.array([r, s])          # (2,)
            pt = Point(float(xq[0]), float(xq[1]))

            # locate transfer cell j that contains/touches xq
            j = None
            for cand in tree.query(pt):
                # cand may be geometry or int depending on shapely version
                if isinstance(cand, (int, np.integer)):
                    j0 = int(cand)
                else:
                    j0 = polyid_to_j.get(id(cand), None)
                    if j0 is None:
                        continue
                if prepared[j0].contains(pt) or prepared[j0].touches(pt):
                    j = j0
                    break
            # robust fallback: barycentric check over triangle vertices of j0
            if j is None:
                for j_try, verts in enumerate(tri_tr):
                    V_inv = V_inv_tr[j_try]
                    lam = V_inv @ np.array([xq[0], xq[1], 1.0])
                    if np.all(lam >= -eps):
                        j = j_try
                        break
            if j is None:
                raise RuntimeError("Quadrature point not found in any transfer cell")

            # ********* FIX 1: evaluate affine poly of cell j directly *********
            # u_tr is (n_tr, 3): [a, b, c] per transfer cell
            a, b0, c0 = u_tr[j, :]
            uq = a * xq[0] + b0 * xq[1] + c0

            # contribution to b with exact weights
            wK = area / 3.0
            lam_coarse = np.array([1.0 - r - s, r, s])  # φ at xq in coarse cell
            b += wK * uq * lam_coarse

        # solve local mass system to get nodal DOFs on this coarse element
        coeffs = M_inv @ b

        # take the coefficient corresponding to node i in this element
        local_i = list(verts_k).index(i)
        u_tgt_nodes[i] = coeffs[local_i]

    # --- reconstruct cell-wise P1 on target from nodal values ---
    # (Exact for constants and P1 on matching meshes)
    p1_tgt = np.empty((g_tgt.num_cells, 3))
    for K, verts in enumerate(tri_tgt):
        V = np.array([[X_tgt[0, verts[0]], X_tgt[1, verts[0]], 1.0],
                      [X_tgt[0, verts[1]], X_tgt[1, verts[1]], 1.0],
                      [X_tgt[0, verts[2]], X_tgt[1, verts[2]], 1.0]])
        rhs = u_tgt_nodes[np.array(verts)]
        a, b0, c0 = np.linalg.solve(V, rhs)
        p1_tgt[K, :] = [a, b0, c0]

    return p1_tgt


# def project_p1(source: pp.GridLike,
#                target: pp.GridLike,
#                u_source: np.ndarray,
#                tol: float = 1e-8) -> np.ndarray:
#     """
#     Do the full two‐step restriction + Scott–Zhang projection:
#       1) build transfer grid
#       2) restrict u_source → u_tr
#       3) project u_tr → u_target
#     """
#     tg = TransferGrid(source, target, tol=tol)
#     u_tr = restrict_to_transfer(tg, u_source)
#     u_tgt = scott_zhang_quasi_interpolant(tg, u_tr)
#     return u_tgt


def _reconstruct_cellwise_on_target(tg: TransferGrid, u_tgt: np.ndarray):
    """
    Given:
      - tg: a TransferGrid with tg.g_target set
      - u_tgt: ndarray of length tg.g_target.num_nodes holding the nodal values
               of the Scott–Zhang interpolant

    Returns:
      - C_tgt: ndarray of shape (n_tgt_cells, 3) with the local P1 coefficients
               [c0, c1, c2] on each target cell, so that
               u(x,y)|_{K} = c0*x + c1*y + c2 on that triangle.
    """
    g_tgt = tg.g_target

    # 1) get rotated 2D coords of target nodes
    tgt_rot = tg._get_rotated_grid(g_tgt)
    Xn = tgt_rot.nodes[:2, :]   # shape (2, n_tgt_nodes)

    # 2) cell->node incidence for target grid
    cn = g_tgt.cell_nodes().tocsc()
    cells = cn.indices.reshape((3, g_tgt.num_cells), order="F").T  # (n_tgt_cells, 3)

    # 3) solve a small Vandermonde per cell
    C_tgt = np.empty((g_tgt.num_cells, 3))
    for k, verts in enumerate(cells):
        # collect the 3 node coordinates
        xy = Xn[:, verts]          # shape (2,3)
        # collect the 3 nodal values
        uvals = u_tgt[verts]       # length-3

        # build 3×3 system: [x y 1]^T * c = uvals
        V = np.vstack((xy, np.ones(3)))  # (3,3)
        # solve V^T * [c0 c1 c2]^T = uvals
        C_tgt[k, :] = np.linalg.solve(V.T, uvals)

    return C_tgt


# def restrict_to_transfer_1d(tl: TransferLine, C_src: np.ndarray) -> np.ndarray:
#     """
#     Source cell-wise P1 on 1D (C_src: n_src × 2 with [a, b] s.t. u(s)=a*s+b)
#     -> transfer cell-wise P1 (n_tr × 2) by sampling at endpoints.
#     """
#     g_tr = tl.transfer
#     t2s = tl.transfer_to_source  # (n_tr × n_src)
#     # endpoints of each transfer cell
#     x = g_tr.nodes[0, :]
#     C_tr = np.empty((g_tr.num_cells, 2))
#     for j in range(g_tr.num_cells):
#         # parent source cell id
#         parents = t2s[j, :].nonzero()[1]
#         if len(parents) != 1:
#             raise RuntimeError(f"Transfer cell {j} has {len(parents)} parents (1D).")
#         K = parents[0]
#         a, b = C_src[K, :]
#         s0, s1 = x[j], x[j + 1]
#         u0, u1 = a * s0 + b, a * s1 + b
#         # P1 on [s0,s1] is determined by two values
#         if abs(s1 - s0) < 1e-16:
#             C_tr[j, :] = [0.0, u0]
#         else:
#             a_loc = (u1 - u0) / (s1 - s0)
#             b_loc = u0 - a_loc * s0
#             C_tr[j, :] = [a_loc, b_loc]
#     return C_tr
#
#
# # --- NEW: 1D Scott–Zhang onto target grid ---
# def scott_zhang_quasi_interpolant_1d(tl: TransferLine, u_tr: np.ndarray) -> np.ndarray:
#     """
#     1D SZ: compute nodal values on the target by local mass solve on one
#     adjacent coarse cell, then reconstruct cell-wise P1 on target.
#     """
#     g_tr  = tl.transfer
#     g_tgt = tl.g_target
#
#     # barycentric on 1D is trivial; we just need values at quadrature points
#     # choose 2-pt Gauss (exact for linears)
#     quad = [(-1/np.sqrt(3), 1.0), (1/np.sqrt(3), 1.0)]  # r in [-1,1], weight 1
#
#     # Build helper: evaluate transfer P1 on point s by locating the transfer cell
#     # that contains s; since tl has uniform connectivity, we can use binary search
#     xt = g_tr.nodes[0, :]
#
#     def eval_u_transfer(s: float) -> float:
#         # locate transfer cell j
#         j = np.searchsorted(xt, s, side="right") - 1
#         j = min(max(j, 0), g_tr.num_cells - 1)
#         a, b = u_tr[j, :]
#         return a * s + b
#
#     # Node values on target via local mass matrix inversion
#     n_tn = g_tgt.num_nodes
#     u_nodes = np.empty(n_tn)
#
#     # target cell intervals
#     xT = g_tgt.nodes[0, :]
#
#     # Precompute, for each node i, pick one adjacent coarse cell:
#     # for endpoints, pick the only adjacent; for interior, pick left cell.
#     node_cell = np.empty(n_tn, dtype=int)
#     node_cell[0] = 0
#     for i in range(1, n_tn - 1):
#         node_cell[i] = i - 1
#     node_cell[-1] = g_tgt.num_cells - 1
#
#     for i in range(n_tn):
#         k = node_cell[i]
#         aT, bT = xT[k], xT[k + 1]
#         h = bT - aT
#         # local P1 basis φ0, φ1 supported on [aT,bT]
#         # mass matrix M = ∫ φ_i φ_j ds = h/6 * [[2,1],[1,2]]
#         M_inv = (6.0 / h) * np.array([[ 2., -1.],
#                                       [-1.,  2.]]) / 3.0  # exact inverse of h/6*[[2,1],[1,2]]
#         # RHS b_i = ∫ u(s) φ_i(s) ds, do 2-pt Gauss on reference → map to [aT,bT]
#         b = np.zeros(2)
#         for xi, w in quad:
#             s = 0.5 * (aT + bT) + 0.5 * h * xi
#             uval = eval_u_transfer(float(s))
#             # shape functions at s: φ0 = (bT - s)/h, φ1 = (s - aT)/h
#             phi0 = (bT - s) / h
#             phi1 = (s - aT) / h
#             b += (0.5 * h) * w * uval * np.array([phi0, phi1])
#         coeffs = M_inv @ b  # [u_i, u_(i+/-1)] on this element
#         # choose the local coefficient corresponding to node i
#         local = 0 if i == k else (1 if i == k + 1 else (0 if i == 0 else 1))
#         u_nodes[i] = coeffs[local]
#
#     # reconstruct cell-wise P1 on target from nodal values
#     C_tgt = np.empty((g_tgt.num_cells, 2))
#     for K in range(g_tgt.num_cells):
#         aT, bT = xT[K], xT[K + 1]
#         u0, u1 = u_nodes[K], u_nodes[K + 1]
#         a_loc = (u1 - u0) / (bT - aT)
#         b_loc = u0 - a_loc * aT
#         C_tgt[K, :] = [a_loc, b_loc]
#
#     return C_tgt
#
#
# def project_p1_1d(
#         source: pp.Grid,
#         target: pp.Grid,
#         u_source: np.ndarray,
#         rotation_matrix: np.ndarray | None = None,
#         tol: float = 1e-10
#     ) -> np.ndarray:
#     """1D pipeline: build TransferLine, restrict, SZ."""
#     tl = TransferLine(source, target, rotation_matrix=rotation_matrix, tol=tol)
#     u_tr = restrict_to_transfer_1d(tl, u_source)
#     u_tgt = scott_zhang_quasi_interpolant_1d(tl, u_tr)
#     return u_tgt

# def project_p1_1d_sz(
#     source: pp.Grid,
#     target: pp.Grid,
#     C_src: np.ndarray,  # shape (n_src_cells, 2): [slope, intercept] in a common frame
#     tol: float = 1e-10,
#     rotation_matrix: np.ndarray | None = None,
#     rotate_source: bool = True,
#     rotate_target: bool = True,
# ) -> np.ndarray:
#     """
#     1D Scott–Zhang (≃ Clément) quasi-interpolant: broken P1 on `source` -> conforming P1 on `target`.
#
#     Outline (robust 1D variant):
#       1) Build a TransferLine (which fixes a common 1D frame).
#       2) For each transfer segment j = [a_j, b_j], integrate u exactly: ∫_j u = 0.5*|j|*(u(a_j)+u(b_j)).
#          (u is the source broken-P1, evaluated using source cell ownership).
#       3) Aggregate those exact integrals to get target cell averages:
#            avg(K) = (sum_j ∫_j u) / |K|
#       4) Nodal values on the target (SZ/Clément):
#            u(x_0) = avg(K_0), u(x_N) = avg(K_{N-1}), interior i: u(x_i) = 0.5*(avg(K_{i-1})+avg(K_i))
#       5) Per target cell K=[a,b], fit P1 from endpoint nodal values with simple stable formula.
#     """
#     from mdnme.utils.transfer_grid import TransferLine
#
#     # --- 1) Shared 1D frame & connectivity via TransferLine ---
#     tl = TransferLine(
#         source,
#         target,
#         tol=tol,
#         rotation_matrix=rotation_matrix,
#         rotate_source=rotate_source,
#         rotate_target=rotate_target,
#     )
#
#     gS, gT, gTr = tl.g_source, tl.g_target, tl.transfer
#
#     T2Tr = tl.target_to_transfer  # (n_tgt x n_tr) CSR; rows list transfer cells inside a target cell
#
#     # Use the transfer line's own break extractors (already sorted & unique in the shared frame)
#     xs  = tl._breaks(gS)   # len = n_src_nodes
#     xt  = tl._breaks(gT)   # len = n_tgt_nodes
#     xtr = tl._breaks(gTr)  # len = n_tr_nodes
#
#     n_src = gS.num_cells
#     n_tgt = gT.num_cells
#     n_tr  = gTr.num_cells
#
#     # Owner source cell (by right-closed convention)
#     def owner_src(xvals: np.ndarray) -> np.ndarray:
#         ids = np.searchsorted(xs, xvals, side="right") - 1
#         return np.clip(ids, 0, len(xs) - 2)
#
#     # Evaluate source broken P1 at arbitrary x in the common frame
#     a_src = C_src[:, 0]      # shape (n_src,)
#     b_src = C_src[:, 1]      # shape (n_src,)
#     def u_src(xvals: np.ndarray) -> np.ndarray:
#         ids = owner_src(xvals)
#         return a_src[ids] * xvals + b_src[ids]
#
#     # --- 2) Exact integrals on transfer segments ---
#     # For j-th transfer cell [xtr[j], xtr[j+1]],
#     # ∫_j u = 0.5*h*(u(a)+u(b)); robust if h<=tol -> treat as 0
#     integ_tr = np.zeros(n_tr, dtype=float)
#     for j in range(n_tr):
#         aJ, bJ = float(xtr[j]), float(xtr[j + 1])
#         h = bJ - aJ
#         if h <= tol:
#             # zero-measure or near-zero: contributes nothing
#             integ_tr[j] = 0.0
#         else:
#             ua = float(u_src(np.array([aJ]))[0])
#             ub = float(u_src(np.array([bJ]))[0])
#             integ_tr[j] = 0.5 * h * (ua + ub)
#
#     # --- 3) Target cell averages: sum intersecting transfer integrals, divide by |K| ---
#     avg_tgt = np.zeros(n_tgt, dtype=float)
#     for k in range(n_tgt):
#         cols = T2Tr.getrow(k).indices  # transfer segments covering target cell k
#         if cols.size == 0:
#             # Should not happen if TL is correct; but keep it safe: set avg=0
#             avg_tgt[k] = 0.0
#         else:
#             aK, bK = float(xt[k]), float(xt[k + 1])
#             hK = bK - aK
#             if hK <= tol:
#                 # degenerate target cell (should not happen); set harmless value
#                 avg_tgt[k] = 0.0
#             else:
#                 avg_tgt[k] = integ_tr[cols].sum() / hK
#
#     # --- 4) Node values (SZ/Clément in 1D) ---
#     #   boundary: use adjacent cell average,
#     #   interior: average of the two neighbors.
#     u_nodes = np.zeros(n_tgt + 1, dtype=float)
#     if n_tgt > 0:
#         u_nodes[0]  = avg_tgt[0]
#         u_nodes[-1] = avg_tgt[-1]
#     if n_tgt > 1:
#         u_nodes[1:-1] = 0.5 * (avg_tgt[:-1] + avg_tgt[1:])
#
#     # --- 5) Build P1 per target cell from endpoint values with a stable closed form ---
#     # Given K=[a,b], ua=u(x=a), ub=u(x=b):
#     # slope = (ub - ua) / (b - a); intercept = ua - slope * a
#     C_tgt = np.zeros((n_tgt, 2), dtype=float)
#     for k in range(n_tgt):
#         aK, bK = float(xt[k]), float(xt[k + 1])
#         hK = bK - aK
#         ua = float(u_nodes[k])
#         ub = float(u_nodes[k + 1])
#         slope = (ub - ua) / hK
#         intercept = ua - slope * aK
#         C_tgt[k, :] = [slope, intercept]
#
#     return C_tgt

# def project_p1_1d_sz(
#     source: pp.Grid,
#     target: pp.Grid,
#     C_src: np.ndarray,  # shape (n_src_cells, 2): [slope, intercept] in a common frame
#     tol: float = 1e-10,
#     rotation_matrix: np.ndarray | None = None,
#     rotate_source: bool = True,
#     rotate_target: bool = True,
# ) -> np.ndarray:
#     """
#     1D P1-preserving projection: broken P1 on `source` -> conforming P1 on `target`.
#
#     We:
#       1) Build a TransferLine (which fixes a common 1D coordinate).
#       2) Define u_src(s) from C_src on the source cells (in that common frame).
#       3) Evaluate u_src at each target node.
#       4) On each target cell K = [a, b], fit u(s) = a_K s + b_K from the two endpoint
#          nodal values (P1-exact if C_src is P1 in the same coordinate).
#     """
#     from mdnme.utils.transfer_grid import TransferLine
#
#     # --- 1) Shared 1D frame via TransferLine ---
#     tl = TransferLine(
#         source,
#         target,
#         tol=tol,
#         rotation_matrix=rotation_matrix,
#         rotate_source=rotate_source,
#         rotate_target=rotate_target,
#     )
#     gS, gT = tl.g_source, tl.g_target
#
#     # Break-points of source/target in the common 1D coordinate
#     xs = tl._breaks(gS)   # len = n_src_nodes
#     xt = tl._breaks(gT)   # len = n_tgt_nodes = n_tgt_cells + 1
#
#     n_src = gS.num_cells
#     n_tgt = gT.num_cells
#
#     a_src = C_src[:, 0]      # (n_src,)
#     b_src = C_src[:, 1]      # (n_src,)
#
#     # --- 2) Owner map and evaluation of broken P1 on source ---
#
#     def owner_src(xvals: np.ndarray) -> np.ndarray:
#         """
#         For each x in xvals, return the index of the source cell whose
#         interval [xs[k], xs[k+1]] contains x (right-closed convention).
#         """
#         ids = np.searchsorted(xs, xvals, side="right") - 1
#         return np.clip(ids, 0, len(xs) - 2)
#
#     def u_src(xvals: np.ndarray) -> np.ndarray:
#         """Evaluate the broken P1 on the source at arbitrary points xvals."""
#         ids = owner_src(xvals)
#         return a_src[ids] * xvals + b_src[ids]
#
#     # --- 3) Target nodal values (sample the broken P1) ---
#     u_nodes = np.empty_like(xt)
#     for i, s in enumerate(xt):
#         u_nodes[i] = float(u_src(np.array([float(s)]))[0])
#
#     # --- 4) Build P1 per target cell from endpoint nodal values ---
#     C_tgt = np.zeros((n_tgt, 2), dtype=float)
#     for k in range(n_tgt):
#         aK, bK = float(xt[k]), float(xt[k + 1])
#         hK = bK - aK
#         if abs(hK) <= tol:
#             # Degenerate cell; just propagate the left nodal value as constant
#             C_tgt[k, :] = [0.0, float(u_nodes[k])]
#         else:
#             ua = float(u_nodes[k])
#             ub = float(u_nodes[k + 1])
#             slope = (ub - ua) / hK
#             intercept = ua - slope * aK
#             C_tgt[k, :] = [slope, intercept]
#
#     return C_tgt

# import numpy as np
# import porepy as pp
# from mdnme.utils.transfer_grid import TransferLine


def restrict_to_transfer_1d(
        tl: TransferLine,
        C_src: np.ndarray,
        tol: float = 1e-12
    ) -> np.ndarray:
    """Restrict a broken 1D P1 field from source -> transfer grid.

    Parameters
    ----------
        tl : TransferLine
             TransferLine object (has .g_source, .transfer, .transfer_to_source).

        C_src : (n_src_cells, 2) ndarray
            Cell-wise P1 coefficients [a, b] on the *source* grid, so that
            u(s)|_K = a*s + b in the common 1D coordinate.

        tol : float
            Tolerance for degenerate segments.

    Returns
    -------
        C_tr : (n_tr_cells, 2) ndarray
            Cell-wise P1 coefficients on the transfer grid.

    """
    g_tr = tl.transfer
    t2s = tl.transfer_to_source.tocsr()   # (n_tr_cells x n_src_cells)

    x_tr = g_tr.nodes[0, :]              # common 1D coordinate of transfer nodes
    n_tr = g_tr.num_cells

    C_tr = np.empty((n_tr, 2))
    for j in range(n_tr):
        # identify parent source cell (TransferLine guarantees one parent)
        parents = t2s[j].indices
        if parents.size != 1:
            raise RuntimeError(
                f"Transfer cell {j} has {parents.size} parents (expected 1)."
            )
        K = parents[0]
        a_src, b_src = C_src[K, :]

        s0, s1 = float(x_tr[j]), float(x_tr[j + 1])
        h = s1 - s0

        if abs(h) <= tol:
            # degenerate; treat as constant on this segment
            u0 = a_src * s0 + b_src
            C_tr[j, :] = [0.0, u0]
        else:
            u0 = a_src * s0 + b_src
            u1 = a_src * s1 + b_src
            a_loc = (u1 - u0) / h
            b_loc = u0 - a_loc * s0
            C_tr[j, :] = [a_loc, b_loc]

    return C_tr


def scott_zhang_quasi_interpolant_1d(
    tl: TransferLine,
    C_tr: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    1D Scott–Zhang quasi–interpolant on the target grid.

    Follows the same spirit as the 2D version:

      1) For each *target* cell K, solve a 2×2 local mass system
         M d = b to obtain the nodal DOFs on that cell.
      2) For each target node i, pick one adjacent cell as its
         patch element and take the corresponding local DOF.
      3) Reconstruct cell-wise P1 from these nodal values.

    Parameters
    ----------
    tl : TransferLine
        TransferLine with .transfer (fine line) and .g_target (coarse line).
    C_tr : (n_tr_cells, 2) ndarray
        Cell-wise P1 coefficients [a, b] on the transfer grid.
    tol : float
        Tolerance for degenerate cells.

    Returns
    -------
    C_tgt : (n_tgt_cells, 2) ndarray
        Cell-wise P1 coefficients [a, b] on the target grid.
    """
    # Retrieve canonical rotation matrix and effective dimension from transfer line
    # Transfer line already knows if we need to rotate node coordinates or not
    # We use that information here
    rot_matrix = tl.rot_matrix
    rotate_source = tl._rotate_source
    rotate_target = tl._rotate_target

    # Retrieve transfer line grid
    g_tr = tl.transfer
    n_tr_cells = g_tr.num_cells
    x_tr = g_tr.nodes[0, :].flatten()  # no-need to rotate

    # Retrieve target grid, and interface side grid.
    g_tgt = tl.g_target
    n_tgt_cells = g_tgt.num_cells
    # We have to rotate the grid
    x_tgt_full = rot_matrix @ g_tgt.nodes
    x_tgt = x_tgt_full[tl.dim_bool].flatten()

    # --- helper: evaluate u on the transfer mesh at arbitrary s ---
    def eval_u_transfer(s: float) -> float:
        # locate transfer cell j containing s (right-closed convention)
        j = np.searchsorted(x_tr, s, side="right") - 1
        j = max(0, min(j, n_tr_cells - 1))
        a, b = C_tr[j, :]
        return a * s + b

    # --- target cell connectivity (each cell has 2 nodes) ---
    cn_tgt = g_tgt.cell_nodes().tocsc()
    # reshape: (n_tgt_cells, 2)
    cells_tgt = cn_tgt.indices.reshape((2, n_tgt_cells), order="F").T

    # node -> cell adjacency
    n2c = g_tgt.cell_nodes().tocsr()

    # --- step 1: local mass solves per target cell K ---
    #
    # For K = [aT, bT], P1 basis:
    #   φ0(s) = (bT - s) / h,   φ1(s) = (s - aT) / h,  h = bT - aT.
    # Mass matrix:
    #   M = (h/6) * [[2,1],[1,2]]
    # Inverse:
    #   M_inv = (2/h) * [[2,-1],[-1,2]]
    #
    # We form RHS:
    #   b_i = ∫_K u(s) φ_i(s) ds, i=0,1
    # via 2-pt Gauss (exact for linear).

    quad = [(-1.0 / np.sqrt(3.0), 1.0),
            ( 1.0 / np.sqrt(3.0), 1.0)]

    local_dofs = np.zeros((n_tgt_cells, 2))

    for K in range(n_tgt_cells):
        n0, n1 = cells_tgt[K]
        aT, bT = float(x_tgt[n0]), float(x_tgt[n1])
        h = bT - aT

        if abs(h) <= tol:
            # degenerate: take constant value at midpoint
            s_mid = 0.5 * (aT + bT)
            val = eval_u_transfer(s_mid)
            local_dofs[K, :] = val
            continue

        # exact M_inv for P1 on [aT, bT]
        M_inv = (2.0 / h) * np.array([[2.0, -1.0],
                                      [-1.0, 2.0]])

        b = np.zeros(2)
        mid = 0.5 * (aT + bT)
        for xi, w in quad:
            # map reference xi ∈ [-1,1] to s ∈ [aT,bT]
            s = mid + 0.5 * h * xi
            uval = eval_u_transfer(float(s))
            # shape functions at s
            phi0 = (bT - s) / h
            phi1 = (s - aT) / h
            # ds = (h/2) dxi
            b += 0.5 * h * w * uval * np.array([phi0, phi1])

        local_dofs[K, :] = M_inv @ b  # [u at node n0, u at node n1]

    # --- step 2: pick patch element for each node & assign nodal value ---
    n_nodes = g_tgt.num_nodes
    u_nodes = np.zeros(n_nodes)

    for i in range(n_nodes):
        adj = n2c[i].indices
        if adj.size == 0:
            raise RuntimeError(f"Target node {i} has no adjacent cell.")

        # Scott–Zhang: pick one adjacent cell as patch.  We choose adj[0].
        K_patch = int(adj[0])
        n0, n1 = cells_tgt[K_patch]

        if i == n0:
            loc = 0
        elif i == n1:
            loc = 1
        else:
            # shouldn't happen if connectivity is consistent
            raise RuntimeError(
                f"Node {i} is not a vertex of its patch cell {K_patch}."
            )

        u_nodes[i] = local_dofs[K_patch, loc]

    # --- step 3: reconstruct cell-wise P1 from nodal values ---
    C_tgt = np.zeros((n_tgt_cells, 2))
    for K in range(n_tgt_cells):
        n0, n1 = cells_tgt[K]
        aT, bT = float(x_tgt[n0]), float(x_tgt[n1])
        ua, ub = float(u_nodes[n0]), float(u_nodes[n1])
        h = bT - aT

        if abs(h) <= tol:
            C_tgt[K, :] = [0.0, ua]
        else:
            slope = (ub - ua) / h
            intercept = ua - slope * aT
            C_tgt[K, :] = [slope, intercept]

    return C_tgt


def project_p1_1d_sz(
    source: pp.Grid,
    target: pp.Grid,
    C_src: np.ndarray,  # (n_src_cells, 2): [slope, intercept] in the common 1D coord
    tol: float = 1e-10,
    rotation_matrix: np.ndarray | None = None,
    dim_bool: np.ndarray | None = None,
    rotate_source: bool = True,
    rotate_target: bool = True,
) -> np.ndarray:
    """
    Full 1D P1 Scott–Zhang projection:

      - Builds a TransferLine(source, target, ...),
      - Restricts source cell-wise P1 -> transfer P1,
      - Applies 1D Scott–Zhang onto the target grid.

    """
    tl = TransferLine(
        source,
        target,
        tol=tol,
        rotation_matrix=rotation_matrix,
        dim_bool=dim_bool,
        rotate_source=rotate_source,
        rotate_target=rotate_target,
    )

    C_tr = restrict_to_transfer_1d(tl, C_src, tol=tol)
    C_tgt = scott_zhang_quasi_interpolant_1d(tl, C_tr, tol=tol)
    return C_tgt

