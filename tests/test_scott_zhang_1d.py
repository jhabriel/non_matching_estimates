import numpy as np
import porepy as pp
import pytest

from mdnme.utils.transfer_grid import TransferLine
from mdnme.utils.primal_projections import scott_zhang_quasi_interpolant_1d


def make_1d(level: int) -> pp.TensorGrid:
    n = 8 * 2**level  # refine by doubling
    x = np.linspace(0.0, 1.0, n + 1).reshape(1, -1)
    g = pp.TensorGrid(x)
    g.compute_geometry()
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

    def u(s): return a2 * s * s + a1 * s + a0
    def du(s): return 2 * a2 * s + a1

    hs, eL2, eH1 = [], [], []

    # Coarse source
    g_src = make_1d(1)  # e.g. 16 cells

    for lev in range(2, 5):  # finer targets: 32, 64, 128 cells
        g_tgt = make_1d(lev)

        # Geometric transfer to get the transfer grid
        tl_geom = TransferLine(
            g_src,
            g_tgt,
            rotation_matrix=np.eye(3),
            dim_bool=np.array([True, False, False]),
            rotate_source=False,
            rotate_target=False,
        )

        # Build exact P1 on the transfer grid by fitting u at segment endpoints
        x = tl_geom.transfer.nodes[0, :]
        C_tr = np.empty((tl_geom.transfer.num_cells, 2))
        for j in range(tl_geom.transfer.num_cells):
            s0, s1 = x[j], x[j + 1]
            u0, u1 = u(s0), u(s1)
            a = (u1 - u0) / (s1 - s0)
            b = u0 - a * s0
            C_tr[j, :] = [a, b]

        # Dummy TL providing a 3D canonical frame compatible with SZ
        class DummyTL:
            def __init__(self, real):
                self.transfer = real.transfer
                self.g_target = real.g_target
                # 3×3 identity rotation (3D-embedded line)
                self.rot_matrix = np.eye(3)
                # one active dimension (x-direction)
                self.dim_bool = np.array([True, False, False], dtype=bool)
                # flags are not used inside scott_zhang_quasi_interpolant_1d,
                # but we define them for completeness
                self._rotate_source = False
                self._rotate_target = False

        tl = DummyTL(tl_geom)

        # Scott–Zhang on the target grid
        C_tgt = scott_zhang_quasi_interpolant_1d(tl, C_tr)

        # Errors via exact 2-point Gauss on each target cell
        xt = g_tgt.nodes[0, :]
        L2_sq, H1_sq = 0.0, 0.0
        for K in range(g_tgt.num_cells):
            aK, bK = xt[K], xt[K + 1]
            h = bK - aK
            # 2-pt Gauss on [-1,1], mapped to [aK,bK]
            for xi, w in [(-1 / np.sqrt(3), 1.0), (1 / np.sqrt(3), 1.0)]:
                s = 0.5 * (aK + bK) + 0.5 * h * xi
                uh = C_tgt[K, 0] * s + C_tgt[K, 1]
                duh = C_tgt[K, 0]
                L2_sq += (0.5 * h) * w * (u(s) - uh) ** 2
                H1_sq += (0.5 * h) * w * (du(s) - duh) ** 2

        # uniform grid: last cell size = h
        hs.append(h)
        eL2.append(np.sqrt(L2_sq))
        eH1.append(np.sqrt(H1_sq))

    hs = np.array(hs)
    eL2 = np.array(eL2)
    eH1 = np.array(eH1)

    # ------------------------------------------------------------------
    # Two regimes:
    # 1) If the operator is "too good" and error is at roundoff level,
    #    we just assert that everything is essentially zero.
    # 2) Otherwise, we enforce the expected h^2 / h rates.
    # ------------------------------------------------------------------
    tol_roundoff_L2 = 1e-12
    tol_roundoff_H1 = 1e-11

    if np.all(eL2 < tol_roundoff_L2) and np.all(eH1 < tol_roundoff_H1):
        # Scott–Zhang is effectively exact for this setup (within roundoff).
        # In that case, convergence *rates* are meaningless; we just check
        # that the errors really are tiny.
        assert np.max(eL2) < tol_roundoff_L2
        assert np.max(eH1) < tol_roundoff_H1
    else:
        # Asymptotic regime: check that we see the expected rates.
        pL2, _ = np.polyfit(np.log(hs), np.log(eL2 + 1e-300), 1)
        pH1, _ = np.polyfit(np.log(hs), np.log(eH1 + 1e-300), 1)
        assert 1.6 <= pL2 <= 2.4, f"L2 rate ~{pL2:.2f} (expected ~2)"
        assert 0.7 <= pH1 <= 1.3, f"H1 rate ~{pH1:.2f} (expected ~1)"
