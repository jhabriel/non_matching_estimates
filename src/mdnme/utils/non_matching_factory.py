"""
Module containing utility functions to generate non-matching grids.
"""

from dataclasses import dataclass

import numpy as np
import porepy as pp


@dataclass
class NMFParams:
    eps: float = 0.08  # max displacement as fraction of local h
    n_blobs: int = 4  # number of smooth “RBF” blobs
    blob_sigma: float = 3.0  # blob width in units of domain diameter / blob_sigma
    seed: int = 0  # RNG seed
    min_angle_deg: float = 20  # soft quality guard (diag only, no flips)
    backtrack: float = 0.5  # scaling for step-back if inversion detected
    max_bt_iter: int = 6  # max backtracking steps


def _face_edges_from_grid(g):
    """Return edges as unique pairs (u,v) from face_nodes; also boundary faces idx."""
    FN = g.face_nodes.tocsr()  # (n_nodes x n_faces)
    # Boundary faces: try common PorePy tags/APIs
    if hasattr(g, "get_boundary_faces"):
        bfaces = np.array(g.get_boundary_faces(), dtype=int)
    elif hasattr(g, "tags") and "domain_boundary_faces" in g.tags:
        bfaces = np.flatnonzero(g.tags["domain_boundary_faces"])
    else:
        # Heuristic: faces with single incident cell
        CF = g.cell_faces.tocsr()
        face_incidence = np.array(np.abs(CF).sum(axis=0)).ravel()
        bfaces = np.flatnonzero(face_incidence == 1)

    n_faces = g.num_faces
    edges = []
    for f in range(n_faces):
        nodes_f = FN[:, f].indices
        if len(nodes_f) == 2:
            u, v = nodes_f
            if u > v:
                u, v = v, u
            edges.append((u, v))
    edges = np.array(list(set(edges)), dtype=int)
    return edges, bfaces


def _boundary_nodes(g):
    edges, bfaces = _face_edges_from_grid(g)
    FN = g.face_nodes.tocsr()
    bnodes = set()
    for f in bfaces:
        nodes_f = FN[:, f].indices
        for n in nodes_f:
            bnodes.add(int(n))
    return np.array(sorted(list(bnodes)), dtype=int)


def _incident_edges_per_node(n_nodes, edges):
    """Adjacency lists and edge lengths for each node."""
    adj = [[] for _ in range(n_nodes)]
    for e_idx, (u, v) in enumerate(edges):
        adj[u].append((e_idx, v))
        adj[v].append((e_idx, u))
    return adj


def _local_h(nodes2d, adj):
    n = nodes2d.shape[1]
    h = np.zeros(n)
    for i in range(n):
        if not adj[i]:
            h[i] = 0.0
            continue
        ds = []
        xi = nodes2d[:, i]
        for _, j in adj[i]:
            ds.append(np.linalg.norm(nodes2d[:, j] - xi))
        h[i] = np.mean(ds) if ds else 0.0
    # Fallback for isolated: use global avg
    if np.any(h == 0):
        mean_h = np.mean(h[h > 0])
        h[h == 0] = mean_h
    return h


def _domain_diameter(nodes2d):
    mins = nodes2d.min(axis=1)
    maxs = nodes2d.max(axis=1)
    return np.linalg.norm(maxs - mins)


def _rbf_displacement(nodes2d, mask_interior, params: NMFParams):
    rng = np.random.default_rng(params.seed)
    n = nodes2d.shape[1]
    diam = _domain_diameter(nodes2d)
    # Choose centers from interior nodes (or all if few)
    idx_pool = np.flatnonzero(mask_interior)
    if idx_pool.size < params.n_blobs:
        idx_pool = np.arange(n)
    centers_idx = rng.choice(idx_pool, size=params.n_blobs, replace=False)
    C = nodes2d[:, centers_idx]  # (2, k)
    # Random directions
    V = rng.normal(size=(2, params.n_blobs))
    V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-15
    # Smooth widths
    sigma = (diam / params.blob_sigma) * np.ones(params.n_blobs)
    # Evaluate field
    X = nodes2d[:, :, None]  # (2, n, 1)
    C3 = C[:, None, :]  # (2, 1, k)
    r2 = np.sum((X - C3) ** 2, axis=0)  # (n, k)
    w = np.exp(-r2 / (2 * sigma[None, :] ** 2))  # (n, k)
    disp = (w @ V.T).T  # (2, n)
    # Normalize to unit max magnitude
    m = np.max(np.linalg.norm(disp, axis=0))
    if m > 0:
        disp /= m
    return disp  # (2, n), unit max


def _min_triangle_angles_deg(nodes2d, cells):
    # cells: (3, n_cells) int
    A = nodes2d[:, cells[0]]
    B = nodes2d[:, cells[1]]
    C = nodes2d[:, cells[2]]
    AB = B - A
    BC = C - B
    CA = A - C
    BA = A - B
    CB = B - C
    AC = C - A

    def angles(U, V):
        num = np.sum(U * V, axis=0)
        den = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) + 1e-15
        cosang = np.clip(num / den, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    aA = angles(AB, AC)  # at A
    aB = angles(BA, BC)  # at B
    aC = angles(CA, CB)  # at C
    amin = np.minimum(np.minimum(aA, aB), aC)
    return amin  # shape (n_cells,)


def _any_inverted(nodes2d, cells):
    A = nodes2d[:, cells[0]]
    B = nodes2d[:, cells[1]]
    C = nodes2d[:, cells[2]]
    # signed area 0.5 * cross( (B-A), (C-A) )
    cross = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    return np.any(cross <= 0)


def build_nonmatching_target(source_grid, params: NMFParams = NMFParams()):
    """
    Returns a PorePy TriangleGrid with identical topology to source_grid
    but interior nodes smoothly perturbed. Boundary is fixed.
    """
    g = source_grid
    if g.dim != 2:
        raise ValueError("This factory currently supports 2D grids.")
    P = (
        g.nodes.copy()
    )  # shape (nd, n_nodes). PorePy often stores (3, n) even for 2D (z=0)
    nd, n = P.shape
    if nd < 2:
        raise ValueError("Grid nodes must have at least 2 coordinates.")
    nodes2d = P[:2, :].copy()

    # Identify boundary vs interior nodes
    bnodes = _boundary_nodes(g)
    mask_boundary = np.zeros(n, dtype=bool)
    mask_boundary[bnodes] = True
    mask_interior = ~mask_boundary

    # Local length scale per node
    edges, _ = _face_edges_from_grid(g)
    adj = _incident_edges_per_node(n, edges)
    h = _local_h(nodes2d, adj)

    # Smooth displacement (unit max), scaled per-node by eps * h
    disp = _rbf_displacement(nodes2d, mask_interior, params)
    scale = params.eps * h
    disp *= scale[None, :]
    disp[:, mask_boundary] = 0.0  # keep boundary exact

    # Backtracking to avoid inverted elements
    cells = g.cells.copy()  # (nodes_per_cell x n_cells) usually (3, m)
    new_nodes2d = nodes2d + disp
    bt_iter = 0
    while _any_inverted(new_nodes2d, cells) and bt_iter < params.max_bt_iter:
        disp *= params.backtrack
        new_nodes2d = nodes2d + disp
        bt_iter += 1

    if _any_inverted(new_nodes2d, cells):
        raise RuntimeError(
            "Could not find a non-inverting displacement; try smaller eps."
        )

    # Soft angle guard (diagnostic)
    amin = _min_triangle_angles_deg(new_nodes2d, cells)
    if np.min(amin) < params.min_angle_deg:
        # Not fatal; warn the caller so they can reduce eps if desired.
        print(
            f"[NMF] Warning: min target angle {np.min(amin):.2f}°"
            f" < {params.min_angle_deg}°"
        )

    # Assemble target grid with same topology
    P_new = P.copy()
    P_new[:2, :] = new_nodes2d
    target_grid = (
        pp.TriangleGrid(P_new[:2, :], cells)
        if nd == 2
        else pp.TriangleGrid(P_new[:2, :], cells)
    )
    return target_grid
