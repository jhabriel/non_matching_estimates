"""Module for testing the prolongation of scalar fields from a source grid onto the
transfer grid."""

import numpy as np
import porepy as pp
import pytest

import mdnme
from mdnme.utils.primal_projections import prolong_to_transfer
from mdnme.utils.transfer_grid import TransferGrid


@pytest.fixture(scope="module")
def coarse_fracture() -> pp.Grid:
    domain = pp.Domain(
        {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    )
    frac = pp.PlaneFracture(
        np.array(
            [
                [0.50, 0.50, 0.50, 0.50],
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.75, 0.75],
            ]
        )
    )
    fn = pp.create_fracture_network([frac], domain)
    mesh_args = {
        "cell_size_boundary": 1.0,
        "cell_size_fracture": 0.1,
        "cell_size_min": 0.02,
    }
    mdg = pp.create_mdg("simplex", mesh_args, fn)
    return mdg.subdomains()[1]


@pytest.fixture(scope="module")
def fine_fracture() -> pp.Grid:
    domain = pp.Domain(
        {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    )
    frac = pp.PlaneFracture(
        np.array(
            [
                [0.50, 0.50, 0.50, 0.50],
                [0.25, 0.75, 0.75, 0.25],
                [0.25, 0.25, 0.75, 0.75],
            ]
        )
    )
    fn = pp.create_fracture_network([frac], domain)
    mesh_args = {
        "cell_size_boundary": 1.0,
        "cell_size_fracture": 0.05,
        "cell_size_min": 0.02,
    }
    mdg = pp.create_mdg("simplex", mesh_args, fn)
    return mdg.subdomains()[1]


def test_constant_field(coarse_fracture, fine_fracture):
    """Checks if a constant field is correctly restricted to the transfer grid."""
    transfer = TransferGrid(coarse_fracture, fine_fracture)

    # Synthetic constant field on the source grid (coarse grid)
    constant_field = np.zeros((coarse_fracture.num_cells, 3))
    constant_field[:, -1] = 1
    transfer_p1 = prolong_to_transfer(transfer, constant_field)
    desired = np.zeros((transfer.transfer.num_cells, 3))
    desired[:, -1] = 1
    np.testing.assert_array_almost_equal(transfer_p1, desired, 10)


def test_local_linear_field(coarse_fracture, fine_fracture):
    """Checks if linear local fields are correctly restricted to the transfer
    grid."""
    """Checks whether a constant field is correctly restricted to the transfer grid."""
    transfer = TransferGrid(coarse_fracture, fine_fracture)

    # Synthetic constant field on the source grid (coarse grid)
    constant_field = np.tile([2, 3, 5], (coarse_fracture.num_cells, 1))
    transfer_p1 = prolong_to_transfer(transfer, constant_field)
    desired = np.tile([2, 3, 5], (transfer.transfer.num_cells, 1))
    np.testing.assert_array_almost_equal(transfer_p1, desired, 10)


@pytest.mark.parametrize(
    "alpha, beta, gamma",
    [
        # pure x‐dependence
        (1.0, 0.0, 0.0),
        # pure y‐dependence
        (0.0, 1.0, 0.0),
        # drop from y=0.25→y=0.75
        (0.0, -2.0, 1.5),
        # diagonal slope
        (1.5, -0.5, 0.2),
        # arbitrary
        (-2.3, 4.7, -1.1),
    ],
)
def test_global_linear_field_param(coarse_fracture, fine_fracture, alpha, beta, gamma):
    """
    Parametrized check that a global P(x,y)=αx+βy+γ on the source grid
    is exactly reconstructed cell‐wise on the transfer grid.
    """
    # 1) Build transfer grid
    tg = TransferGrid(coarse_fracture, fine_fracture)

    # 2) Build C_src by fitting P to each coarse cell
    src_rot = mdnme.RotatedGrid(coarse_fracture)
    Xc = src_rot.nodes[:2, :]
    cn = coarse_fracture.cell_nodes().tocsc()
    cells = cn.indices.reshape((3, coarse_fracture.num_cells), order="F").T

    C_src = np.empty((coarse_fracture.num_cells, 3))
    for k, verts in enumerate(cells):
        xy = Xc[:, verts]  # (2×3)
        uvals = alpha * xy[0, :] + beta * xy[1, :] + gamma
        V = np.vstack((xy, np.ones(3)))  # (3×3)
        C_src[k, :] = np.linalg.solve(V.T, uvals)

    # 3) Restrict to transfer cells
    C_tr = prolong_to_transfer(tg, C_src)

    # 4) Evaluate at transfer‐cell centroids
    cc = tg.transfer.cell_centers
    x_tr = cc[0, :]
    y_tr = cc[1, :]
    u_tr = C_tr[:, 0] * x_tr + C_tr[:, 1] * y_tr + C_tr[:, 2]

    # 5) True P at centroids
    u_true = alpha * x_tr + beta * y_tr + gamma

    # 6) Assert exactness
    np.testing.assert_allclose(u_tr, u_true, atol=1e-12, rtol=0)
