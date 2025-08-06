"""Module for testing the restriction of scalar fields from a source grid onto the
transfer grid."""

import numpy as np
import porepy as pp
import pytest

from mdnme.utils.transfer_grid import TransferGrid
from mdnme.utils.primal_projections import restrict_to_transfer


@pytest.fixture(scope="module")
def coarse_fracture() -> pp.Grid:
    domain = pp.Domain(
        {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
    )
    frac = pp.PlaneFracture(np.array([[0.50, 0.50, 0.50, 0.50],
                                      [0.25, 0.75, 0.75, 0.25],
                                      [0.25, 0.25, 0.75, 0.75]]))
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
    frac = pp.PlaneFracture(np.array([[0.50, 0.50, 0.50, 0.50],
                                      [0.25, 0.75, 0.75, 0.25],
                                      [0.25, 0.25, 0.75, 0.75]]))
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
    transfer_p1 = restrict_to_transfer(transfer, constant_field)
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
    transfer_p1 = restrict_to_transfer(transfer, constant_field)
    desired = np.tile([2, 3, 5], (transfer.transfer.num_cells, 1))
    np.testing.assert_array_almost_equal(transfer_p1, desired, 10)


def test_global_linear_field(coarse_fracture, fine_fracture):
    """
    Impose a linear drop from u=1 at y=0.25 down to u=0 at y=0.75 on the source grid,
    then check that cell‐wise restriction onto the transfer grid reproduces the same
    line.
    """
    transfer = TransferGrid(coarse_fracture, fine_fracture)

    # Build global P(x,y) = α·x + β·y + γ such that
    #   P(y=0.25) = 1  and  P(y=0.75) = 0, independent of x.
    β = (0.0 - 1.0) / (0.75 - 0.25)    # = -2.0
    α = 0.0                            # no x‐dependence
    γ = 1.0 - β * 0.25                 # = 1.5

    # Source cell‐wise coefficients: same for all source cells
    C_src = np.tile([α, β, γ], (coarse_fracture.num_cells, 1))

    # Restrict to transfer cells
    C_tr = restrict_to_transfer(transfer, C_src)

    # Evaluate each transfer‐cell polynomial at its centroid
    cc = transfer.transfer.cell_centers   # shape (>=2, n_tr_cells)
    x_tr = cc[0, :]
    y_tr = cc[1, :]
    u_tr = C_tr[:, 0] * x_tr + C_tr[:, 1] * y_tr + C_tr[:, 2]

    # True line value at those centroids
    u_true = α * x_tr + β * y_tr + γ

    # Should match to machine precision
    np.testing.assert_allclose(u_tr, u_true, atol=1e-12, rtol=0)
