"""
Module containing functionality to project P1 potentials from source to target grids.

The two main projection steps are:

    (1) Restriction from the source grid onto the transfer grid.
    (2) Projection from transfer grid onto the target grid using the Scott–Zhang
        quasi‐interpolator.

"""

import numpy as np
import porepy as pp

from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree

from mdnme.utils.transfer_grid import TransferGrid


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


def project_p1(source: pp.GridLike,
               target: pp.GridLike,
               u_source: np.ndarray,
               tol: float = 1e-8) -> np.ndarray:
    """
    Do the full two‐step restriction + Scott–Zhang projection:
      1) build transfer grid
      2) restrict u_source → u_tr
      3) project u_tr → u_target
    """
    tg = TransferGrid(source, target, tol=tol)
    u_tr = restrict_to_transfer(tg, u_source)
    u_tgt = scott_zhang_quasi_interpolant(tg, u_tr)
    return u_tgt


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
