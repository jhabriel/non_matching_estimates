"""Library of mixed-dimensional grids.

Mainly for use in tests. Other usage should be covered by the model_geometries.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union, cast

import numpy as np

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d


def benchmark_3d_case_2(
    refinement_level: Literal[0, 1, 2] = 0,
) -> tuple[pp.MixedDimensionalGrid, FractureNetwork3d]:
    """
    Create a mixed-dimensional grid for the geometry of case 2 from [1].

    Note:
        The mixed-dimensional grid is created by reading a `geo` file, so there is no
        direct way of prescribing meshing arguments.

    Reference:
        [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D.,
        Keilegavlen, E., ... & Zulian, P. (2021). Verification benchmarks for
        single-phase flow in three-dimensional fractured porous media. Advances in Water
        Resources, 147, 103759.

    Parameters:
        refinement_level: An integer denoting the level of refinement. Use `0` to
            generate a mixed-dimensional grid with approximately 500 3D cells, `1` for
            4K 3D cells, `2` for 32K 3D cells.

    Returns:
        Tuple containing a:

            :obj:`~pp.MixedDimensionalGrid`:
                The grid for the specified refinement level.

            :obj:`~pp.FractureNetwork3d`:
                The fracture network.

    """
    # Sanity check on input argument
    if refinement_level not in [0, 1, 2]:
        raise NotImplementedError("Refinement level not available.")

    # Get directory pointing to the `geo` file
    abs_path = Path(__file__)
    benchmark_path = abs_path.parent / "gmsh_file_library" / "benchmark_3d_case_2"
    if refinement_level == 0:
        full_path = benchmark_path / f"mesh500.geo"
    elif refinement_level == 1:
        full_path = benchmark_path / f"mesh4k.geo"
    elif refinement_level == 2:
        full_path = benchmark_path / f"mesh32k.geo"
    else:\


    num_cells = [30, 140, 350, 500][refinement_level]
    full_path = benchmark_path / f"mesh{num_cells}k.geo"

    # Set file permissions. This turned out to be important for GH actions.
    full_path.chmod(777)

    # Create mixed-dimensional grid
    mdg = pp.fracture_importer.dfm_from_gmsh(full_path, dim=3)

    # Also import fracture network
    fracture_network_path = benchmark_path / "fracture_network.csv"
    # Set file permissions. This turned out to be important for GH actions.
    fracture_network_path.chmod(777)

    network = pp.fracture_importer.network_3d_from_csv(fracture_network_path)

    return mdg, network
