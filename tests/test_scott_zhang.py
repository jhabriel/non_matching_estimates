# tests/test_sz_quadratic_convergence.py
import numpy as np
import porepy as pp
import pytest
import mdnme

from porepy.grids.refinement import GridSequenceFactory
from mdnme.utils.transfer_grid import TransferGrid, coarse_fine_or_build
from mdnme.utils.primal_projections import scott_zhang_quasi_interpolant
from mdnme.utils.primal_projections import prolong_to_transfer
from mdnme.utils.grid_utils import refine_grid
from mdnme.utils.grid_rotation import assign_canonical_rotations


# ---------- helpers ----------
def tri_edges_max_length(coords):  # coords shape (2,3)
    e0 = np.linalg.norm(coords[:, 1] - coords[:, 0])
    e1 = np.linalg.norm(coords[:, 2] - coords[:, 1])
    e2 = np.linalg.norm(coords[:, 0] - coords[:, 2])
    return max(e0, e1, e2)


def dunavant_deg5():  # exact up to degree 5 (good for L2 of quadratic^2, deg 4)
    # barycentric (r,s) with weights on reference triangle
    pts = np.array([
        [1 / 3, 1 / 3, 9 / 40],
        [0.0597158717, 0.4701420641, 0.0661970764],
        [0.4701420641, 0.0597158717, 0.0661970764],
        [0.4701420641, 0.4701420641, 0.0661970764],
        [0.1012865073, 0.7974269853, 0.0629695903],
        [0.7974269853, 0.1012865073, 0.0629695903],
        [0.1012865073, 0.1012865073, 0.0629695903],
    ])
    return pts[:, :2], pts[:, 2]


def tri_3pt():  # degree-2 exact (perfect for H1 error of quadratics)
    pts = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    return pts, w


def quad_u_and_grad(a, b, c, d, e, f):
    def u(x, y): return a * x * x + b * x * y + c * y * y + d * x + e * y + f

    def gx(x, y): return 2 * a * x + b * y + d

    def gy(x, y): return b * x + 2 * c * y + e

    return u, gx, gy


# ---------- fixtures ----------

@pytest.fixture(scope="module")
def grid_sequence():
    # fracture-in-a-unit-cube, same as before
    domain = pp.Domain(
        {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1})
    frac = pp.PlaneFracture(np.array([
        [0.50, 0.50, 0.50, 0.50],
        [0.25, 0.75, 0.75, 0.25],
        [0.25, 0.25, 0.75, 0.75],
    ]))
    fn = pp.create_fracture_network([frac], domain)

    mesh_args = {  # coarsish base; factory will refine
        "mesh_size_bound": 0.4,
        "mesh_size_frac": 0.4,
        "mesh_size_min": 0.01,
    }
    params = {"mode": "nested", "num_refinements": 4, "mesh_param": mesh_args}
    factory = GridSequenceFactory(fn, params)
    mdgs = list(factory)
    # pick the 2D fracture subdomain from each MDG
    levels = [mdg.subdomains()[1] for mdg in mdgs]
    return levels  # coarse -> ... -> finest


def sz_on_same_grid(grid: pp.Grid, p1_broken: np.ndarray) -> np.ndarray:
    """Make a broken P1 field H¹-conforming on the *same* grid via SZ."""
    tg = TransferGrid(grid, grid)                  # same mesh -> same frame
    p_on_tg = prolong_to_transfer(tg, p1_broken)  # evaluate on transfer
    return scott_zhang_quasi_interpolant(tg, p_on_tg)


def test_matching_equals_nonmatching_after_h1_conformity():
    # 1) pick a representative 2D grid (e.g., a fracture or a mortar side)
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    mdg = pp.create_mdg("simplex", {"cell_size": 0.15}, pp.create_fracture_network([], domain))
    g = mdg.subdomains(dim=2)[0]

    # 2) fabricate a *broken* P1 (random per-cell coefficients)
    rng = np.random.default_rng(42)
    p_broken = rng.standard_normal((g.num_cells, 3))

    # 3) make it H¹-conforming on G (Oswald/SZ)
    p_h1 = sz_on_same_grid(g, p_broken)

    # 4) matching path on a matching target: identity
    p_match = p_h1.copy()

    # 5) non-matching pipeline, but with source==target (should act as identity on H¹)
    tg = TransferGrid(g, g)
    p_on_tg = prolong_to_transfer(tg, p_h1)
    p_nonmatch = scott_zhang_quasi_interpolant(tg, p_on_tg)

    # 6) they must coincide (up to roundoff)
    np.testing.assert_allclose(p_nonmatch, p_match, rtol=5e-12, atol=5e-14)


@pytest.mark.parametrize("coeffs", [
    # u(x,y) = x^2
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    # u(x,y) = y^2
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    # u(x,y) = x y
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    # u(x,y) = x^2 + y^2 + x
    (1.0, 0.0, 1.0, 1.0, 0.0, 0.0),
])
def test_sz_quadratic_convergence(grid_sequence, coeffs):
    """
    Expect ||u - I_h u||_{L2} ~ O(h^2), |u - I_h u|_{H1} ~ O(h) for quadratics.
    """
    a, b, c, d, e, f = coeffs
    u, gx, gy = quad_u_and_grad(a, b, c, d, e, f)

    levels = grid_sequence
    finest = levels[-1]  # use finest as "source" just to define transfer grid

    # storage for (h, L2, H1) at several target levels
    hs, eL2s, eH1s = [], [], []

    for g_tgt in levels[:-1]:  # use every level except the finest as target
        tg = TransferGrid(finest, g_tgt)

        # --- build u_tr on the transfer mesh nodes by analytic sampling ---
        C_tr = fit_p1_on_grid(tg.transfer, u)

        # --- SZ v2: get cell-wise P1 on target ---
        C_tgt = scott_zhang_quasi_interpolant(tg, C_tr)

        # --- compute errors on target grid ---
        tgt_rot = mdnme.RotatedGrid(g_tgt)
        Xn = tgt_rot.nodes[:2, :]
        cn = g_tgt.cell_nodes().tocsc()
        tri = cn.indices.reshape((3, g_tgt.num_cells), order="F").T

        # mesh size h = max edge length
        h_level = 0.0
        for verts in tri:
            coords = Xn[:, verts]
            h_level = max(h_level, tri_edges_max_length(coords))
        hs.append(h_level)

        # L2 error via Dunavant degree-5 (exact for deg-4 integrands)
        qpts, qw = dunavant_deg5()
        L2_sq = 0.0
        for k, verts in enumerate(tri):
            coords = Xn[:, verts]
            p0, p1, p2 = coords[:, 0], coords[:, 1], coords[:, 2]
            A = np.column_stack((p1 - p0, p2 - p0))
            area = abs(np.linalg.det(A)) * 0.5
            c0, c1, c2 = C_tgt[k, :]
            for (r, s), w in zip(qpts, qw):
                xq = p0 + A.dot([r, s])
                u_exact = u(xq[0], xq[1])
                u_h = c0 * xq[0] + c1 * xq[1] + c2
                L2_sq += w * area * (u_exact - u_h) ** 2
        eL2s.append(np.sqrt(L2_sq))

        # H1 error: |grad(u - u_h)| via 3-point degree-2 (exact)
        q3, w3 = tri_3pt()
        H1_sq = 0.0
        for k, verts in enumerate(tri):
            coords = Xn[:, verts]
            p0, p1, p2 = coords[:, 0], coords[:, 1], coords[:, 2]
            A = np.column_stack((p1 - p0, p2 - p0))
            area = abs(np.linalg.det(A)) * 0.5
            # grad(u_h) is constant on the cell:
            c0, c1, _ = C_tgt[k, :]
            for (r, s), w in zip(q3, w3):
                xq = p0 + A.dot([r, s])
                dux = gx(xq[0], xq[1]) - c0
                duy = gy(xq[0], xq[1]) - c1
                H1_sq += w * area * (dux * dux + duy * duy)
        eH1s.append(np.sqrt(H1_sq))

    # --- check observed rates via least-squares on log-log ---
    hs = np.array(hs)
    eL2s = np.array(eL2s)
    eH1s = np.array(eH1s)
    eps = 1e-300
    p_L2, _ = np.polyfit(np.log(hs), np.log(eL2s + eps), 1)
    p_H1, _ = np.polyfit(np.log(hs), np.log(eH1s + eps), 1)

    # slopes should be ~2 (L2) and ~1 (H1)
    assert 1.6 <= p_L2 <= 2.4, f"L2 slope off: {p_L2:.2f}"
    assert 0.7 <= p_H1 <= 1.3, f"H1 slope off: {p_H1:.2f}"


def fit_p1_on_grid(grid: pp.Grid, u_fn) -> np.ndarray:
    """Return cell-wise P1 (a,b,c) fitting u at the three vertices of each triangle."""
    rot = mdnme.RotatedGrid(grid)
    X   = rot.nodes[:2, :]
    cn  = grid.cell_nodes().tocsc()
    tri = cn.indices.reshape((3, grid.num_cells), order="F").T

    C = np.empty((grid.num_cells, 3))
    for k, verts in enumerate(tri):
        xy    = X[:, verts]                      # 2×3
        uvals = u_fn(xy[0, :], xy[1, :])         # length-3
        V     = np.vstack((xy, np.ones(3)))      # 3×3, rows [x;y;1]
        C[k, :] = np.linalg.solve(V.T, uvals)
    return C



def test_nested_vs_geometric_SZ_projection():
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    fn = pp.create_fracture_network([], domain)
    mdg = pp.create_mdg("simplex", {"cell_size": 0.12}, fn)
    G0 = mdg.subdomains()[0]
    G1, _ = refine_grid(G0.copy())  # fine target

    # random per-cell P1 coefficients on source grid (G0)
    rng = np.random.default_rng(7)
    C_src = rng.standard_normal((G0.num_cells, 3))

    # geometric path
    tg_geo = TransferGrid(G0, G1, tol=1e-10)
    C_tr_geo = prolong_to_transfer(tg_geo, C_src)
    C_tgt_geo = scott_zhang_quasi_interpolant(tg_geo, C_tr_geo)

    # nested path
    M = coarse_fine_or_build(G0, G1, tol=1e-10)
    tg_fast = TransferGrid.from_nested(G0, G1, coarse_fine=M, tol=1e-10)
    C_tr_fast = prolong_to_transfer(tg_fast, C_src)
    C_tgt_fast = scott_zhang_quasi_interpolant(tg_fast, C_tr_fast)

    np.testing.assert_allclose(C_tgt_fast, C_tgt_geo, rtol=0, atol=1e-12)
