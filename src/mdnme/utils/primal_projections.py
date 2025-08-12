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


from scipy.sparse import csr_matrix

def restrict_to_transfer(tg, C_src):
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



def scott_zhang_on_transfer(tg: TransferGrid, u_tr: np.ndarray) -> np.ndarray:
    """
    Scott–Zhang quasi‐interpolation from u_tr on tg.transfer to P1 values
    on tg.g_target.

    Parameters
    ----------
    tg : TransferGrid
      Must have tg.transfer and tg.transfer_to_target and tg.g_target set.
    u_tr : array_like, shape (tg.transfer.num_nodes,)
      Nodal values on the transfer grid.

    Returns
    -------
    u_tgt : np.ndarray, shape (tg.g_target.num_nodes,)
      Nodal P1 values on the target grid.
    """
    g_tr = tg.transfer
    g_tgt = tg.g_target
    t2tgt = tg.transfer_to_target.tocsr()  # (n_tr_cells × n_tgt_cells)

    # 1) cell→node connectivity
    cn_tr = g_tr.cell_nodes().tocsc()
    tri_tr = cn_tr.indices.reshape((3, g_tr.num_cells), order="F").T
    cn_tgt = g_tgt.cell_nodes().tocsc()
    tri_tgt = cn_tgt.indices.reshape((3, g_tgt.num_cells), order="F").T

    # 2) Precompute per‐target‐cell M⁻¹ and V⁻¹
    X_tgt = g_tgt.nodes[:2, :]
    M_inv = [None] * g_tgt.num_cells
    V_inv = [None] * g_tgt.num_cells
    for k, verts in enumerate(tri_tgt):
        coords = X_tgt[:, verts]
        area = abs(np.linalg.det(np.vstack((coords[:, 1] - coords[:, 0],
                                            coords[:, 2] - coords[:, 0])).T)) * 0.5
        M = (area / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        M_inv[k] = np.linalg.inv(M)
        V = np.vstack((coords, np.ones(3)))
        V_inv[k] = np.linalg.inv(V)

    # 3) Map each target cell → list of transfer‐cells inside it
    tr_in_tgt = {k: t2tgt[:, k].nonzero()[0] for k in range(g_tgt.num_cells)}

    # 4) Quadrature rule
    quad = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])

    # 5) Node → cell adjacency on target
    n2c = g_tgt.cell_nodes().tocsr()

    # 6) Evaluate
    X_tr = g_tr.node_coords[:2, :]
    u_tgt = np.zeros(g_tgt.num_nodes)
    for i in range(g_tgt.num_nodes):
        adj = n2c[i].nonzero()[1]
        if not adj:
            raise RuntimeError(f"Target node {i} has no adjacent cell")
        k = adj[0]
        verts_k = tri_tgt[k]
        loc_i = list(verts_k).index(i)

        b = np.zeros(3)
        for j in tr_in_tgt[k]:
            tri_j = tri_tr[j]
            coords_j = X_tr[:, tri_j]
            uvals = u_tr[tri_j]
            area_j = abs(np.linalg.det(
                np.vstack((coords_j[:, 1] - coords_j[:, 0],
                           coords_j[:, 2] - coords_j[:, 0])).T)) * 0.5
            w = area_j / 3.0
            for xi in quad:
                lambda_j = np.array([xi[0], xi[1], 1 - xi.sum()])
                xq = coords_j.dot(lambda_j)
                uq = uvals.dot(lambda_j)
                lambda_c = V_inv[k].dot(np.append(xq, 1.0))[:3]
                b += w * uq * lambda_c

        coeffs = M_inv[k].dot(b)
        u_tgt[i] = coeffs[loc_i]

    return u_tgt

def scott_zhang_quasi_interpolant(tg, u_tr):
    """
    Robust Scott–Zhang using coarse-cell quadrature + transfer-mesh point
    evaluations. Exact on P1.

    Returns cell-wise P1 coefficients on target.
    """
    g_tr  = tg.transfer
    g_tgt = tg.g_target

    # 1) Precompute coarse-cell data
    cn_tgt   = g_tgt.cell_nodes().tocsc()
    tri_tgt  = cn_tgt.indices.reshape((3, g_tgt.num_cells), order="F").T
    tgt_rot  = tg._get_rotated_grid(g_tgt)
    X_tgt    = tgt_rot.nodes[:2, :]

    coarse_data = []
    for verts in tri_tgt:
        p0, p1, p2 = X_tgt[:, verts[0]], X_tgt[:, verts[1]], X_tgt[:, verts[2]]
        A = np.column_stack((p1 - p0, p2 - p0))
        area = abs(np.linalg.det(A)) * 0.5
        M_inv = (1.0/area) * np.array([[ 9, -3, -3],[-3, 9, -3],[-3, -3, 9]])
        coarse_data.append((verts, p0, A, M_inv, area))

    # 2) Build transfer-mesh STRtree + barycentric helpers
    cn_tr   = g_tr.cell_nodes().tocsc()
    tri_tr  = cn_tr.indices.reshape((3, g_tr.num_cells), order="F").T
    tr_rot  = tg._get_rotated_grid(g_tr)
    X_tr    = tr_rot.nodes[:2, :]

    # Polygons and map
    tr_polygons = [Polygon(X_tr[:, verts].T) for verts in tri_tr]
    poly_to_j   = {poly: j for j, poly in enumerate(tr_polygons)}
    tree        = STRtree(tr_polygons)
    prepared    = [prep(p) for p in tr_polygons]

    # Inverses for barycentric in transfer cells
    V_inv_tr = []
    for verts in tri_tr:
        V = np.vstack((X_tr[:, verts], np.ones(3)))
        V_inv_tr.append(np.linalg.inv(V))

    # 3) Prepare storage
    n_tgt_nodes = g_tgt.num_nodes
    u_tgt_nodes = np.empty(n_tgt_nodes)
    n2c = g_tgt.cell_nodes().tocsr()
    quadrature = [(1/6,1/6),(2/3,1/6),(1/6,2/3)]

    # 4) Loop over target nodes
    for i in range(n_tgt_nodes):
        adj = n2c[i].nonzero()[1]
        if adj.size == 0:
            raise RuntimeError(f"Target node {i} has no adjacent cell")
        k = adj[0]
        verts_k, p0, A, M_inv, area = coarse_data[k]

        # Build RHS b
        b = np.zeros(3)
        for r, s in quadrature:
            xq = p0 + A.dot([r, s])
            pt = Point(*xq)

            # Locate transfer cell j
            j = None
            for cand in tree.query(pt):
                if isinstance(cand, (int, np.integer)):
                    j0 = int(cand)
                else:
                    j0 = poly_to_j[cand]
                if prepared[j0].contains(pt) or prepared[j0].touches(pt):
                    j = j0
                    break
            if j is None:
                raise RuntimeError("Quadrature point not found in any transfer cell")

            # Evaluate u_tr at xq via barycentric in j
            lam_tr = V_inv_tr[j].dot([xq[0], xq[1], 1.0])[:3]
            uq     = u_tr[tri_tr[j]].dot(lam_tr)

            w = area/3.0
            lam_coarse = np.array([1 - r - s, r, s])
            b += w * uq * lam_coarse

        coeffs = M_inv.dot(b)
        local_i = list(verts_k).index(i)
        u_tgt_nodes[i] = coeffs[local_i]

    # 5) Reconstruct cell-wise and return
    return _reconstruct_cellwise_on_target(tg, u_tgt_nodes)

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
    u_tgt = scott_zhang_on_transfer(tg, u_tr)
    return u_tgt

def _reconstruct_cellwise_on_target(tg, u_tgt):
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
        C_tgt[k,:] = np.linalg.solve(V.T, uvals)

    return C_tgt
