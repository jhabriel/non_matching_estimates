import numpy as np
import porepy as pp
import pytest

from mdnme.utils.transfer_grid import TransferLine
from mdnme.utils.primal_projections import (
    prolong_to_transfer_1d,
    scott_zhang_quasi_interpolant_1d,
    project_p1_1d_sz,
)


def make_tensor_grid_1d(breaks):
    # breaks: sorted 1D array of nodes
    x = np.asarray(breaks, dtype=float).reshape(1, -1)
    g = pp.TensorGrid(x)
    g.compute_geometry()
    return g


@pytest.fixture(scope="module")
def g_src():
    # nonmatching: 5 cells on [0, 1]
    return make_tensor_grid_1d(np.linspace(0.0, 1.0, 6))


@pytest.fixture(scope="module")
def g_tgt():
    # nonmatching: 7 cells on [0, 1]
    return make_tensor_grid_1d(np.linspace(0.0, 1.0, 8))


def _make_transfer_line(g_src, g_tgt):
    """Helper: build a TransferLine with a trivial 3D canonical frame."""
    return TransferLine(
        g_src,
        g_tgt,
        rotation_matrix=np.eye(3),
        dim_bool=np.array([True, False, False], dtype=bool),
        rotate_source=False,
        rotate_target=False,
    )


def test_constant_exact(g_src, g_tgt):
    tl = _make_transfer_line(g_src, g_tgt)

    # u(s) = 3.14 on the source
    C_src = np.zeros((g_src.num_cells, 2))
    C_src[:, 1] = 3.14

    C_tr = prolong_to_transfer_1d(tl, C_src)

    # evaluate on transfer midpoints
    x = tl.transfer.nodes[0, :]
    mids = 0.5 * (x[:-1] + x[1:])
    u_tr = C_tr[:, 0] * mids + C_tr[:, 1]

    np.testing.assert_allclose(u_tr, 3.14, rtol=0.0, atol=1e-13)


def test_linear_exact(g_src, g_tgt):
    # u(s) = a s + b must be exact everywhere
    a, b = -2.3, 0.7
    C_src = np.tile([a, b], (g_src.num_cells, 1))

    # full pipeline TL ➜ SZ ➜ target, with explicit 3D canonical frame
    C_tgt = project_p1_1d_sz(
        g_src,
        g_tgt,
        C_src,
        rotation_matrix=np.eye(3),
        dim_bool=np.array([True, False, False], dtype=bool),
        rotate_source=False,
        rotate_target=False,
    )

    xt = g_tgt.nodes[0, :]
    mids = 0.5 * (xt[:-1] + xt[1:])
    u_tgt = C_tgt[:, 0] * mids + C_tgt[:, 1]

    np.testing.assert_allclose(u_tgt, a * mids + b, rtol=0.0, atol=1e-13)


def test_identity_on_matching(g_src):
    # source == target; SZ over TL should act like identity for P1
    C_src = np.array([[1.2, -0.4]] * g_src.num_cells)

    C_tgt = project_p1_1d_sz(
        g_src,
        g_src,
        C_src,
        rotation_matrix=np.eye(3),
        dim_bool=np.array([True, False, False], dtype=bool),
        rotate_source=False,
        rotate_target=False,
    )

    np.testing.assert_allclose(C_tgt, C_src, rtol=0.0, atol=1e-13)


def test_sz_stability_random(g_src, g_tgt):
    # random broken P1 on source → TL → SZ; ensure it returns finite values
    rng = np.random.default_rng(123)
    C_src = rng.standard_normal((g_src.num_cells, 2))

    tl = _make_transfer_line(g_src, g_tgt)
    C_tr = prolong_to_transfer_1d(tl, C_src)
    C_tgt = scott_zhang_quasi_interpolant_1d(tl, C_tr)

    assert C_tgt.shape == (g_tgt.num_cells, 2)
    assert np.isfinite(C_tgt).all()
