import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps

from mdnme.utils.grid_utils import refine_grid
from mdnme.utils.transfer_grid import (
    TransferGrid,
    coarse_fine_or_build,
    permute_transfer_columns,
    transfer_permutation_by_centroids,
)


@pytest.fixture(scope="module")
def base_subdomain() -> pp.Grid:
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    fn = pp.create_fracture_network([], domain)
    mdg = pp.create_mdg("simplex", {"cell_size": 0.1}, fn)
    return mdg.subdomains()[0]


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


domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
fn = pp.create_fracture_network([], domain)
mdg = pp.create_mdg("simplex", {"cell_size": 0.1}, fn)
sd0 = mdg.subdomains()[0]


def test_source_equal_to_target():
    """Test whether a transfer grid is identical to source and target grids,
    when source and target grids are identical"""
    transfer_grid = TransferGrid(sd0, sd0)
    src = transfer_grid.g_source
    tgt = transfer_grid.g_target
    tg = transfer_grid.transfer

    # Same number of cells, faces and nodes
    assert tg.num_cells == src.num_cells and tg.num_cells == tgt.num_cells
    assert tg.num_faces == src.num_faces and tg.num_faces == tgt.num_faces
    assert tg.num_nodes == src.num_nodes and tg.num_nodes == tgt.num_nodes

    # Same domain volume
    np.testing.assert_approx_equal(src.cell_volumes.sum(), tg.cell_volumes.sum(), 7)
    np.testing.assert_approx_equal(tgt.cell_volumes.sum(), tg.cell_volumes.sum(), 7)


def test_target_refined_wrt_source():
    """Test transfer grid generation in the case of a globally red-refined target grid
    created from a source grid."""
    source_grid = sd0.copy()
    target_grid, _ = refine_grid(source_grid)  # global red-refinement
    tgo = TransferGrid(source_grid, target_grid)  # create transfer grid obj
    transfer_grid = tgo.transfer  # retrieve transfer grid

    # In this case, the transfer grid must be identical to the target grid
    assert transfer_grid.num_cells == target_grid.num_cells
    assert transfer_grid.num_faces == target_grid.num_faces
    assert transfer_grid.num_nodes == target_grid.num_nodes

    # Also test the volume
    np.testing.assert_approx_equal(
        target_grid.cell_volumes.sum(), transfer_grid.cell_volumes.sum(), 7
    )

    # Check connectivity and area overlap

    # This test checks whether the area of a source grid cell matches with the sum of
    # the areas of associated transfer cells
    src2tra = tgo.source_to_transfer
    src2tra_scaled = src2tra @ sps.diags(transfer_grid.cell_volumes)
    area = src2tra_scaled.sum(axis=1).A1
    np.testing.assert_allclose(area, source_grid.cell_volumes, rtol=1e-8, atol=1e-8)

    # This test checks whether the area of a target grid matches with the sum of
    # the areas of associated transfer cells
    tgt2tra = tgo.target_to_transfer
    tgt2tra_scaled = tgt2tra @ sps.diags(transfer_grid.cell_volumes)
    area = tgt2tra_scaled.sum(axis=1).A1
    np.testing.assert_allclose(area, target_grid.cell_volumes, rtol=1e-8, atol=1e-8)

    # Check that each transfer cell maps to exactly one source cell
    tra2src = tgo.transfer_to_source
    src_counts = tra2src.sum(axis=1).A1
    assert np.all(src_counts == 1)

    # Check that each transfer cell maps to exactly one target cell
    tra2tgt = tgo.transfer_to_target
    tgt_counts = tra2tgt.sum(axis=1).A1
    assert np.all(tgt_counts == 1)


def test_source_refined_wrt_target():
    """Test transfer grid generation in the case of a globally red-refined source grid
    created from a target grid."""
    target_grid = sd0.copy()
    source_grid, _ = refine_grid(target_grid)  # global red-refinement
    tgo = TransferGrid(source_grid, target_grid)  # create transfer grid obj
    transfer_grid = tgo.transfer  # retrieve transfer grid

    # In this case, the transfer grid must be identical to the target grid
    assert transfer_grid.num_cells == source_grid.num_cells
    assert transfer_grid.num_faces == source_grid.num_faces
    assert transfer_grid.num_nodes == source_grid.num_nodes

    # Also test the volume
    np.testing.assert_approx_equal(
        source_grid.cell_volumes.sum(), transfer_grid.cell_volumes.sum(), 7
    )

    # Check connectivity and area overlap

    # This test checks whether the area of a source grid cell matches with the sum of
    # the areas of associated transfer cells
    src2tra = tgo.source_to_transfer
    src2tra_scaled = src2tra @ sps.diags(transfer_grid.cell_volumes)
    area = src2tra_scaled.sum(axis=1).A1
    np.testing.assert_allclose(area, source_grid.cell_volumes, rtol=1e-8, atol=1e-8)

    # This test checks whether the area of a target grid matches with the sum of
    # the areas of associated transfer cells
    tgt2tra = tgo.target_to_transfer
    tgt2tra_scaled = tgt2tra @ sps.diags(transfer_grid.cell_volumes)
    area = tgt2tra_scaled.sum(axis=1).A1
    np.testing.assert_allclose(area, target_grid.cell_volumes, rtol=1e-8, atol=1e-8)

    # Check that each transfer cell maps to exactly one source cell
    tra2src = tgo.transfer_to_source
    src_counts = tra2src.sum(axis=1).A1
    assert np.all(src_counts == 1)

    # Check that each transfer cell maps to exactly one target cell
    tra2tgt = tgo.transfer_to_target
    tgt_counts = tra2tgt.sum(axis=1).A1
    assert np.all(tgt_counts == 1)


def test_perturbed_target_wrt_source():
    """
    Checks whether the transfer grid is correctly constructed in the case when
    the internal nodes of the target grid are perturbed wrt the source grid.

    Note:
        Perturbing nodes results in triangles of very bad shape. Thus, this test
        is also a good stress test for the TransferGrid class.

    """
    source_grid = sd0.copy()

    # Target grid is source grid but the internal nodes are horizontally and
    # vertically perturbed by 25% of the average cell diameter of the grid
    target_grid = sd0.copy()
    int_nodes = target_grid.get_internal_nodes()
    mean_diam = np.mean(target_grid.cell_diameters())
    target_grid.nodes[0][int_nodes] = target_grid.nodes[0][int_nodes] + 0.2 * mean_diam
    target_grid.nodes[1][int_nodes] = target_grid.nodes[1][int_nodes] + 0.2 * mean_diam
    target_grid.compute_geometry()

    # Create transfer grid
    tgo = TransferGrid(source_grid, target_grid)
    transfer_grid = tgo.transfer

    # Check that the total area is preserved
    assert np.isclose(
        source_grid.cell_volumes.sum(),
        transfer_grid.cell_volumes.sum(),
        1e-3,
        1e-4,
    )
    assert np.isclose(
        target_grid.cell_volumes.sum(),
        transfer_grid.cell_volumes.sum(),
        1e-3,
        1e-4,
    )

    # Check connectivity and area overlap

    # This test checks whether the area of a source grid cell matches with the sum of
    # the areas of associated transfer cells
    src2tra = tgo.source_to_transfer
    src2tra_scaled = src2tra @ sps.diags(transfer_grid.cell_volumes)
    area = src2tra_scaled.sum(axis=1).A1
    np.testing.assert_allclose(area, source_grid.cell_volumes, rtol=1e-3, atol=1e-4)

    # This test checks whether the area of a target grid matches with the sum of
    # the areas of associated transfer cells
    tgt2tra = tgo.target_to_transfer
    tgt2tra_scaled = tgt2tra @ sps.diags(transfer_grid.cell_volumes)
    area = tgt2tra_scaled.sum(axis=1).A1
    np.testing.assert_allclose(area, target_grid.cell_volumes, rtol=1e-3, atol=1e-4)

    # Check that each transfer cell maps to exactly one source cell
    tra2src = tgo.transfer_to_source
    src_counts = tra2src.sum(axis=1).A1
    assert np.all(src_counts == 1)

    # Check that each transfer cell maps to exactly one target cell
    tra2tgt = tgo.transfer_to_target
    tgt_counts = tra2tgt.sum(axis=1).A1
    assert np.all(tgt_counts == 1)


@pytest.mark.parametrize("fracture", ["coarse_fracture", "fine_fracture"])
def test_embedded_identical_source_and_target_grids(fracture, request):
    """
    Checks transfer grid generation for a single identical fracture grid.
    """
    src = request.getfixturevalue(fracture).copy()
    tgt = src.copy()
    tfo = TransferGrid(src, tgt)
    tg = tfo.transfer

    # Same number of cells, faces and nodes
    assert tg.num_cells == src.num_cells and tg.num_cells == tgt.num_cells
    assert tg.num_faces == src.num_faces and tg.num_faces == tgt.num_faces
    assert tg.num_nodes == src.num_nodes and tg.num_nodes == tgt.num_nodes

    # Same domain volume
    assert np.isclose(src.cell_volumes.sum(), tg.cell_volumes.sum(), 1e-3, 1e-4)
    assert np.isclose(tgt.cell_volumes.sum(), tg.cell_volumes.sum(), 1e-3, 1e-4)


@pytest.mark.parametrize("fracture", ["coarse_fracture", "fine_fracture"])
def test_embedded_perturbed_source_and_target(fracture, request):

    src = request.getfixturevalue(fracture).copy()

    tgt = src.copy()
    tol = 1e-8  # geometric tolerance
    y = tgt.nodes[1]  # y-coordinates
    z = tgt.nodes[2]  # z-coordinates
    # Since the fracture is fully embedded, we need to manually identify the
    # "boundary" and "internal" nodes
    mask_y = np.isclose(y, 0.25, atol=tol) | np.isclose(y, 0.75, atol=tol)
    mask_z = np.isclose(z, 0.25, atol=tol) | np.isclose(z, 0.75, atol=tol)
    pseudo_bnd_nodes = np.nonzero(mask_y | mask_z)[0]
    pseudo_int_nodes = np.setdiff1d(np.arange(tgt.num_nodes), pseudo_bnd_nodes)
    mean_diam = np.mean(tgt.cell_diameters())  # get the average cell-diameter
    shift = 0.5 * mean_diam  # shift half a mean diameter
    tgt.nodes[1][pseudo_int_nodes] += shift  # perturb in y-axis
    tgt.nodes[2][pseudo_int_nodes] += shift  # perturb in z-axis
    tgt.compute_geometry()  # recompute geometry

    tfo = TransferGrid(src, tgt)
    tg = tfo.transfer

    # Volume preservation
    # TODO: There is some important mismatch here, check whether this is indeed a bug
    #  or some geometric thing
    assert np.isclose(src.cell_volumes.sum(), tg.cell_volumes.sum(), 1e-3, 1e-3)
    assert np.isclose(tgt.cell_volumes.sum(), tg.cell_volumes.sum(), 1e-3, 1e-3)

    # Uniqueness / connectivity sanity
    src_counts = tfo.transfer_to_source.sum(axis=1).A1
    tgt_counts = tfo.transfer_to_target.sum(axis=1).A1
    assert np.all(src_counts == 1)
    assert np.all(tgt_counts == 1)


def test_nested_fastpath_equals_geometric_when_target_refined():
    # base grid
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    fn = pp.create_fracture_network([], domain)
    mdg = pp.create_mdg("simplex", {"cell_size": 0.15}, fn)
    G0 = mdg.subdomains()[0]
    # refine target globally (fine target)
    G1, _ = refine_grid(G0.copy())
    G1.compute_geometry()

    # geometric path
    tg_geo = TransferGrid(G0, G1, tol=1e-9)
    # nested path (requires mapping fine×coarse)
    M = coarse_fine_or_build(G0, G1, tol=1e-9)  # (n_fine x n_coarse)
    tg_fast = TransferGrid.from_nested(G0, G1, coarse_fine=M, tol=1e-9, name="fast")

    # transfer is the fine mesh (up to ordering)
    assert tg_fast.transfer.num_cells == G1.num_cells
    assert tg_fast.transfer.num_nodes == G1.num_nodes
    np.testing.assert_allclose(
        tg_fast.transfer.cell_volumes.sum(),
        G1.cell_volumes.sum(),
        rtol=0,
        atol=1e-12,
    )

    # align geometric matrices to fast by transfer-cell permutation
    perm = transfer_permutation_by_centroids(tg_fast, tg_geo)
    s2t_geo_aligned = permute_transfer_columns(tg_geo.source_to_transfer, perm)
    t2s_geo_aligned = tg_geo.transfer_to_source[perm, :]

    # incidence matrices should agree
    np.testing.assert_array_equal(
        s2t_geo_aligned.todense(), tg_fast.source_to_transfer.todense()
    )
    np.testing.assert_array_equal(
        t2s_geo_aligned.todense(), tg_fast.transfer_to_source.todense()
    )


def test_nested_fastpath_equals_geometric_when_source_refined():
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    fn = pp.create_fracture_network([], domain)
    mdg = pp.create_mdg("simplex", {"cell_size": 0.15}, fn)
    G0 = mdg.subdomains()[0]
    # refine source globally (fine source)
    G1, _ = refine_grid(G0.copy())
    G1.compute_geometry()

    # geometric path
    tg_geo = TransferGrid(G1, G0, tol=1e-9)

    # fast path (mapping again via coarse_fine_or_build)
    M = coarse_fine_or_build(G1, G0, tol=1e-9)  # still returns (fine x coarse)
    tg_fast = TransferGrid.from_nested(G1, G0, coarse_fine=M, tol=1e-9, name="fast")

    # transfer is the fine mesh (up to ordering)
    assert tg_fast.transfer.num_cells == G1.num_cells
    assert tg_fast.transfer.num_nodes == G1.num_nodes
    np.testing.assert_allclose(
        tg_fast.transfer.cell_volumes.sum(), G1.cell_volumes.sum(), rtol=0, atol=1e-12
    )

    # align geometric matrices to fast by transfer-cell permutation
    perm = transfer_permutation_by_centroids(tg_fast, tg_geo)
    s2t_geo_aligned = permute_transfer_columns(tg_geo.source_to_transfer, perm)
    t2s_geo_aligned = tg_geo.transfer_to_source[perm, :]

    np.testing.assert_array_equal(
        s2t_geo_aligned.todense(), tg_fast.source_to_transfer.todense()
    )
    np.testing.assert_array_equal(
        t2s_geo_aligned.todense(), tg_fast.transfer_to_source.todense()
    )


def test_nested_fastpath_equals_geometric_when_source_equal_target():
    domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
    fn = pp.create_fracture_network([], domain)
    mdg = pp.create_mdg("simplex", {"cell_size": 0.15}, fn)
    G0 = mdg.subdomains()[0]
    G1 = G0.copy()

    # geometric path
    tg_geo = TransferGrid(G1, G0, tol=1e-9)

    # fast path (mapping again via coarse_fine_or_build)
    M = coarse_fine_or_build(G1, G0, tol=1e-9)  # still returns (fine x coarse)
    tg_fast = TransferGrid.from_nested(G1, G0, coarse_fine=M, tol=1e-9, name="fast")

    # transfer is the fine mesh (up to ordering)
    assert tg_fast.transfer.num_cells == G1.num_cells
    assert tg_fast.transfer.num_nodes == G1.num_nodes
    np.testing.assert_allclose(
        tg_fast.transfer.cell_volumes.sum(), G1.cell_volumes.sum(), rtol=0, atol=1e-12
    )

    # align geometric matrices to fast by transfer-cell permutation
    perm = transfer_permutation_by_centroids(tg_fast, tg_geo)
    s2t_geo_aligned = permute_transfer_columns(tg_geo.source_to_transfer, perm)
    t2s_geo_aligned = tg_geo.transfer_to_source[perm, :]

    np.testing.assert_array_equal(
        s2t_geo_aligned.todense(), tg_fast.source_to_transfer.todense()
    )
    np.testing.assert_array_equal(
        t2s_geo_aligned.todense(), tg_fast.transfer_to_source.todense()
    )
