"""Module for testing the correct construction of the internal boundary grid."""
import mdnme
import porepy as pp
import numpy as np
import pytest


from mdnme.utils.internal_boundary_grid import InternalBoundaryGrid
from mdnme.utils.transfer_grid import TransferGrid
from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import (
    manu_incomp_fluid,
    manu_incomp_solid,
    )

@pytest.fixture(scope="module")
def material_constants() -> dict:
    """Define material constants used throughout the tests."""
    solid_constants = pp.SolidConstants(**manu_incomp_solid)
    fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
    return {"solid": solid_constants, "fluid": fluid_constants}


@pytest.mark.parametrize("refine_mortar", [False, True])
def tests_ibg_construction_single_frac(
        material_constants,
        refine_mortar,
) -> None:

    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "non_matching": True,
        "refine_fracture": False,
        "refine_mortar": refine_mortar,
        "times_to_export": [],  # Supress outputs for tests
    }
    setup = VarelaJNumSetup3D(params)
    pp.run_time_dependent_model(setup, {})

    # Build the internal boundary grid
    mdg = setup.mdg
    sd_high = mdg.subdomains(dim=3)[0]
    intf = mdg.interfaces(dim=2)[0]
    ibg = InternalBoundaryGrid(intf, sd_high)

    for side in ibg.sides():
        P_side = ibg.mortar_to_side(side)
        mg_sidegrid = ibg.mortar_side_grid(side)
        ibg_sidegrid = ibg.ibg_side_grid(side)
        faces = ibg.high_faces(side)
        parent = ibg.parent_face_of_cell(side)

        # shapes/indices are consistent
        assert P_side.shape[0] == mg_sidegrid.num_cells
        assert parent.shape[0] == ibg_sidegrid.num_cells
        assert faces.ndim == 1 and faces.dtype.kind in "iu"

        # if this side has any mortar cells covered by high faces, IBG shouldn't be
        # empty
        if faces.size > 0:
            assert ibg_sidegrid.num_cells > 0

        # Sanity check on partitions
        P_blocks = [ibg.mortar_to_side(s) for s in ibg.sides()]
        P_terms = [Pi.T @ Pi for Pi in P_blocks]  # each is (n_mortar, n_mortar)
        # diagonal on its block
        Ipart = sum(P_terms)  # should be identity on mortar cells
        np.testing.assert_allclose(Ipart.diagonal(), np.ones(intf.num_cells))

        # build the Transfer grid use for projections
        TG_hi2side = TransferGrid(
            g_source=ibg_sidegrid,
            g_target=mg_sidegrid,
            rotation_matrix=ibg.rotation_matrix,
            tol=1e-8,
        )

        assert TG_hi2side.transfer.num_cells >= 0
        assert TG_hi2side.source_to_transfer.shape == (ibg_sidegrid.num_cells,
                                                       TG_hi2side.transfer.num_cells)
        assert TG_hi2side.transfer_to_target.shape == (TG_hi2side.transfer.num_cells,
                                                       mg_sidegrid.num_cells)
