"""Module containing geometry-related mixins for the third numerical example."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast, Literal, Tuple

import numpy as np
import porepy as pp

from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from mdnme.utils.nested_refinement import GeoNestedRefinementFactory

# ---------- small helpers ----------

def _stem_for_refinement_level(refinement_level: Literal[0, 1, 2, 3]) -> str:
    """Map refinement_level -> base stem used in both .geo and .msh names."""
    if refinement_level == 0:
        return "mesh30k"
    elif refinement_level == 1:
        return "mesh140k"
    elif refinement_level == 2:
        return "mesh350k"
    elif refinement_level == 3:
        return "mesh500k"
    raise ValueError("Refinement level not supported. Use 0, 1, 2, or 3.")


def _paths_for_level(
        refinement_level: Literal[0, 1, 2, 3],
        folder: str = "grids"
    ) -> Tuple[Path, Path, Path, str]:
    """
    Returns (geo_path, msh_path, csv_path, out_stem)
    geo_path: <stem>.geo (expected at project root or folder root; see below)
    msh_path: <folder>/<stem>_nonmatch.msh
    csv_path: <folder>/fracture_network.csv
    out_stem: <folder>/<stem> (used by factory; it writes <out_stem>_<k>.msh)
    """
    stem = _stem_for_refinement_level(refinement_level)
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    # where we read/write the non-matching mesh
    msh_path = folder_path / f"{stem}_nonmatch.msh"
    # where we keep the fracture network CSV
    csv_path = folder_path / "fracture_network.csv"

    # .geo can be either in the folder or in the project root—check both
    geo_local = folder_path / f"{stem}.geo"
    geo_root = Path(f"{stem}.geo")
    geo_path = geo_local if geo_local.exists() else geo_root

    # out_stem for Gmsh writes (factory will create <out_stem>_k.msh files)
    out_stem = str(folder_path / stem)
    return geo_path, msh_path, csv_path, out_stem


def create_mdg_from_msh_file(refinement_level: Literal[0, 1, 2, 3]):
    """
    Load a pre-generated non-matching grid (.msh) and its fracture network (.csv).

    Returns:
        mdg: pp.MixedDimensionalGrid
        fn:  FractureNetwork3d
    """
    _, msh_path, csv_path, _ = _paths_for_level(refinement_level)
    if not msh_path.exists():
        raise FileNotFoundError(f"Missing non-matching msh: {msh_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing fracture network CSV: {csv_path}")

    mdg = pp.fracture_importer.dfm_from_gmsh(str(msh_path), dim=3)
    fn = pp.fracture_importer.network_3d_from_csv(str(csv_path), dim=3)
    return mdg, fn


# ---------- main geometry mixin ----------

class GeometryNonMatching(pp.PorePyModel):
    """Define Geometry as specified in Section 5.3 of the benchmark study [1]."""

    # def set_geometry(self) -> None:
    #     """Create mixed-dimensional grid and fracture network (matching or non-matching)."""
    #
    #     ref_lvl: Literal[0, 1, 2, 3] = self.params.get("refinement_level", 0)
    #     non_matching: bool = self.params.get("non_matching", False)
    #
    #     if non_matching:
    #         # Try to reuse an existing .msh (and fracture_network.csv).
    #         geo_path, msh_path, csv_path, out_stem = _paths_for_level(ref_lvl)
    #
    #         if msh_path.exists() and csv_path.exists() and 2 > 3:
    #             print('\t Found existing mesh file. Reading .msh file and creating '
    #                   'mdg...')
    #             # Fast path: just read the saved grid+network
    #             self.mdg, self.fracture_network = create_mdg_from_msh_file(ref_lvl)
    #             print('\t Done reading .msh file and creating mdg.')
    #
    #         else:
    #             print('Generating nonmatching grid...')
    #             # We need to generate the non-matching grid once via factory:
    #             #  - start from the .geo corresponding to ref_lvl
    #             #  - globally refine once (nested)
    #             #  - replace all lower-dim subdomains
    #             #  - write the resulting non-matching .msh into msh_path
    #             if not geo_path.exists():
    #                 raise FileNotFoundError(
    #                     f"Missing {geo_path.name}. Place it at project root or in {geo_path.parent}."
    #                 )
    #
    #             dim = 3
    #             num_refinements = 1
    #             factory = GeoNestedRefinementFactory(
    #                 src_path=str(geo_path),
    #                 dim=dim,
    #                 num_refinements=num_refinements,
    #                 out_stem=out_stem,  # will emit <out_stem>_0.msh, <out_stem>_1.msh, ...
    #             )
    #
    #             mdg_coarse = None
    #             mdg_fine = None
    #             for i, mdg in enumerate(factory):
    #                 if i == 0:
    #                     mdg_coarse = mdg
    #                 else:
    #                     mdg_fine = mdg
    #             if mdg_coarse is None or mdg_fine is None:
    #                 raise RuntimeError("Nested refinement factory did not yield two levels.")
    #
    #             # Replace all lower-dim subdomains with those from the refined one
    #             sd_map = {}
    #             for sd_co, sd_fi in zip(mdg_coarse.subdomains(), mdg_fine.subdomains()):
    #                 assert sd_co.dim == sd_fi.dim
    #                 if sd_co.dim in [2]:
    #                     sd_map[sd_co] = sd_fi
    #
    #             # intf_map = {}
    #             # for intf_co, intf_fi in zip(mdg_coarse.interfaces(),
    #             #                          mdg_fine.interfaces()):
    #             #     assert intf_co.dim == intf_fi.dim
    #             #     if intf_co.dim == 1:
    #             #         intf_map[intf_co] = intf_fi
    #             #
    #             mdg_coarse.replace_subdomains_and_interfaces(sd_map=sd_map)
    #
    #             # Persist the result for future runs
    #             # NOTE: GmshWriter expects gmsh to be initialized in its own flow,
    #             # so we reuse dfm export via Gmsh interface helper
    #             # A robust path is to write using porepy’s gmsh writer through the network,
    #             # but here we already have an mdg: we can use the built-in exporter:
    #             #pp.fracs.gmsh_interface.write_mdg_to_gmsh(mdg_coarse, str(msh_path))
    #
    #             # Also persist the fracture network if not present
    #             if not csv_path.exists():
    #                 # Rebuild the reference network using the benchmark helper,
    #                 # then export it. This keeps CSV consistent with the mdg domain.
    #                 # (If you already have a curated CSV, just place it in the folder.)
    #                 _, fn_ref = benchmark_3d_case_3(refinement_level=ref_lvl)
    #                 # Safe-guard: fn_ref is a FractureNetwork3d
    #                 if not isinstance(fn_ref, FractureNetwork3d):
    #                     raise TypeError("benchmark_3d_case_3 did not return a 3D fracture network.")
    #                 pp.fracture_importer.to_csv(fn_ref, str(csv_path))
    #
    #             # adopt the in-memory mdg/net
    #             self.mdg = mdg_coarse.copy()
    #             self.fracture_network = pp.fracture_importer.network_3d_from_csv(str(csv_path), dim=3)
    #
    #             print('Done generating nonmatching grid.')
    #
    #     else:
    #         # Matching path: create the standard benchmark grid+network
    #         self.mdg, self.fracture_network = benchmark_3d_case_3(
    #             refinement_level=ref_lvl
    #         )
    #
    #     # Bookkeeping: dim, domain, fractures
    #     self.nd: int = self.mdg.dim_max()
    #     self._domain = cast(pp.Domain, self.fracture_network.domain)
    #     self._fractures = self.fracture_network.fractures
    #
    #     # Projections and canonical rotations
    #     pp.set_local_coordinate_projections(self.mdg)
    #
    #     # Wells (unchanged)
    #     self.set_well_network()
    #     if len(self.well_network.wells) > 0:
    #         assert isinstance(self.fracture_network, FractureNetwork3d)
    #         pp.compute_well_fracture_intersections(
    #             self.well_network, self.fracture_network
    #         )
    #         self.well_network.mesh(self.mdg)

    def set_geometry(self) -> None:

        # Whether the mdg is non-matching or not
        non_matching: bool = self.params.get("non_matching", False)

        # Read the fracture network directly from the csv file
        fn = pp.fracture_importer.network_3d_from_csv("grids/fracture_network.csv")
        fn.impose_external_boundary()  # needed to set the bounding box

        # Get grid type and meshing arguments
        grid_type = self.params.get("grid_type", "simplex")
        meshing_args = self.params.get('meshing_arguments', {"cell_size": 0.25})


        # If it is matching, produce the mdg in the usual way
        if not non_matching or non_matching:
            mdg = pp.create_mdg(
                grid_type=grid_type,  # type[ignore]
                fracture_network=fn,
                meshing_args=meshing_args,
            )
        else:

            # Retrieve the target mesh size and create a DFN mdg
            # h = meshing_args["cell_size"]
            # mdg_dfn = pp.create_mdg(
            #     grid_type=grid_type,
            #     fracture_network=fn,
            #     meshing_args={"cell_size": h/4},
            #     dfn=True,
            # )
            h = meshing_args["cell_size"]
            mdg_fine = pp.create_mdg(
                grid_type=grid_type,
                fracture_network=fn,
                meshing_args={"cell_size": h},
                dfn=True,
            )
            # Create a coarse mdg
            mdg = pp.create_mdg(
                grid_type=grid_type,
                fracture_network=fn,
                meshing_args={"cell_size": h},
            )

            # Prepare mapping dictionary to replace grids
            sd_map = {}
            for sd, sd_fine in zip(mdg.subdomains(dim=2), mdg_fine.subdomains(dim=2)):
                assert sd.dim == sd_fine.dim
                sd_map[sd] = sd_fine

            for sd, sd_fine in zip(mdg.subdomains(dim=1), mdg_fine.subdomains(dim=1)):
                assert sd.dim == sd_fine.dim
                sd_map[sd] = sd_fine

            # intf_map = {}
            # for intf, intf_fine in zip(mdg.interfaces(dim=2), mdg_fine.interfaces(
            #         dim=2)):
            #     assert intf.dim == intf_fine.dim
            #     intf_map[intf] = intf_fine

            # Replace grids
            mdg.replace_subdomains_and_interfaces(sd_map=sd_map)

        # Finally, set mdg and fracture network as a public attribute
        self.fracture_network = fn
        self.mdg = mdg

        # Update local projections
        pp.set_local_coordinate_projections(mdg)

        # Bookkeeping: dim, domain, fractures
        self.nd: int = self.mdg.dim_max()
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Projections and canonical rotations
        pp.set_local_coordinate_projections(self.mdg)

        # Wells (unchanged)
        self.set_well_network()
        if len(self.well_network.wells) > 0:
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            self.well_network.mesh(self.mdg)

