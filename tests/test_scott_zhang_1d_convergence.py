import numpy as np
import porepy as pp
import pytest

from mdnme.utils.transfer_grid import TransferLine
from mdnme.utils.primal_projections import (
    prolong_to_transfer_1d,
    scott_zhang_quasi_interpolant_1d,
)

def make_1d(level):
    n = 8 * 2**level  # refine by doubling
    x = np.linspace(0.0, 1.0, n+1).reshape(1, -1)
    g = pp.TensorGrid(x); g.compute_geometry()
    return g

@pytest.mark.parametrize("coeffs", [
    # u(s) = s^2
    (1.0, 0.0, 0.0),
    # u(s) = s^2 + s
    (1.0, 1.0, 0.0),
    # u(s) = s^2 - 3 s + 2
    (1.0, -3.0, 2.0),
])
def test_sz_1d_quadratic_rates(coeffs):
    a2, a1, a0 = coeffs
    def u(s): return a2*s*s + a1*s + a0
    def du(s): return 2*a2*s + a1

    hs, eL2, eH1 = [], [], []
    # Use a coarse source and project to finer targets so the transfer grid refines
    # with the target (and the quadratic interpolation error shrinks)
    g_src = make_1d(1)  # coarse (e.g., 16 cells)
    for lev in range(2, 5):  # targets finer than source: 32, 64, 128 cells
        g_tgt = make_1d(lev)
        # Build transfer & fit the exact quadratic on the (source, target)-dependent
        # transfer grid
        tl = TransferLine(g_src, g_tgt)
        # construct exact C_tr by fitting u on each transfer cell endpoints
        x = tl.transfer.nodes[0, :]
        C_tr = np.empty((tl.transfer.num_cells, 2))
        for j in range(tl.transfer.num_cells):
            s0, s1 = x[j], x[j+1]
            u0, u1 = u(s0), u(s1)
            a = (u1 - u0) / (s1 - s0)
            b = u0 - a * s0
            C_tr[j,:] = [a, b]

        C_tgt = scott_zhang_quasi_interpolant_1d(tl, C_tr)

        # errors via exact 2-point Gauss
        xt = g_tgt.nodes[0, :]
        L2_sq, H1_sq = 0.0, 0.0
        for K in range(g_tgt.num_cells):
            aK, bK = xt[K], xt[K+1]
            h = bK - aK
            # 2-pt Gauss on [-1,1], map to [aK,bK]
            for xi, w in [(-1/np.sqrt(3), 1.0), (1/np.sqrt(3), 1.0)]:
                s = 0.5*(aK+bK) + 0.5*h*xi
                uh  = C_tgt[K,0]*s + C_tgt[K,1]
                duh = C_tgt[K,0]
                L2_sq += (0.5*h)*w * (u(s)  - uh )**2
                H1_sq += (0.5*h)*w * (du(s) - duh)**2
        hs.append(h); eL2.append(np.sqrt(L2_sq)); eH1.append(np.sqrt(H1_sq))

    hs = np.array(hs); eL2 = np.array(eL2); eH1 = np.array(eH1)
    pL2, _ = np.polyfit(np.log(hs), np.log(eL2 + 1e-300), 1)
    pH1, _ = np.polyfit(np.log(hs), np.log(eH1 + 1e-300), 1)
    assert 1.6 <= pL2 <= 2.4, f"L2 rate ~{pL2:.2f} (expected ~2)"
    assert 0.7 <= pH1 <= 1.3, f"H1 rate ~{pH1:.2f} (expected ~1)"
