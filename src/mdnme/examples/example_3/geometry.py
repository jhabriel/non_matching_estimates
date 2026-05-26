"""Module containing geometry-related mixins for the third numerical example."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple, cast

import porepy as pp
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.fracs.fracture_network_3d import FractureNetwork3d

from mdnme.utils.nested_refinement import GeoNestedRefinementFactory

# Absolute path to the grids/ subdirectory next to this file
_GRIDS = Path(__file__).parent / "grids"


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
    out_folder: str = "grids",
) -> Tuple[Path, Path, Path, str]:
    """Return (geo_path, msh_path, csv_path, out_stem) for a refinement level.

    geo_path  — source .geo file (always inside grids/)
    msh_path  — where the non-matching .msh will be written (out_folder)
    csv_path  — fracture network CSV (always inside grids/)
    out_stem  — basename passed to GeoNestedRefinementFactory
    """
    stem = _stem_for_refinement_level(refinement_level)
    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    geo_path = _GRIDS / f"{stem}.geo"
    msh_path = out_path / f"{stem}_nonmatch.msh"
    csv_path = _GRIDS / "fracture_network.csv"
    out_stem = str(out_path / stem)
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

    mdg = pp.fracture_importer.dfm_from_gmsh(msh_path, dim=3)
    fn = pp.fracture_importer.network_from_csv(str(csv_path), dim=3)
    return mdg, fn


# ---------- main geometry mixin ----------


class GeometryNonMatching(pp.PorePyModel):
    """Define Geometry as specified in Section 5.3 of the benchmark study [1]."""

    def set_geometry(self) -> None:

        # Whether the mdg is non-matching or not
        non_matching: bool = self.params.get("non_matching", False)

        # Read the fracture network directly from the csv file
        fn = pp.fracture_importer.network_from_csv(str(_GRIDS / "fracture_network.csv"))

        # Get grid type and meshing arguments
        grid_type = self.params.get("grid_type", "simplex")
        meshing_args = self.params.get("meshing_arguments", {"cell_size": 0.25})

        # If it is matching, produce the mdg in the usual way
        if not non_matching:

            print("Running model using matching grids.")

            # Decide whether to mesh from geo or from fracture network
            from_geo = self.params.get("matching_from_geo", True)

            if not from_geo:

                print("Meshing using mesh parameters.")
                mdg_coarse = pp.create_mdg(
                    grid_type=grid_type,  # type:ignore
                    fracture_network=fn,
                    meshing_args=meshing_args,
                )
            else:  # generate from geo

                print("Meshing using geo files.")
                mdg_coarse, _ = benchmark_3d_case_3(refinement_level=0)

        else:

            print("Running model using non-matching grids")

            # Get refinement strategy
            ref_stgy = self.params.get("refinement", "unstructured")

            if ref_stgy == "unstructured":  # do unstructured refinement

                print("Refinement strategy: unstructured")

                # Retrieve the target mesh size and create a DFN mdg
                h = meshing_args["cell_size"]

                # Create a coarse mdg
                mdg_coarse = pp.create_mdg(
                    grid_type=grid_type,
                    fracture_network=fn,
                    meshing_args={"cell_size": h},
                )

                # Create a fine mdg
                mdg_fine = pp.create_mdg(
                    grid_type=grid_type,  # type:ignore
                    fracture_network=fn,
                    meshing_args={"cell_size": h / 2},
                    # dfn=True,
                )

            elif ref_stgy == "nested":  # do nested refinement

                print("Refinement strategy: nested")

                # Create a nested refinement (one-level) of the whole mdg
                dim = 3
                num_refinements = 1
                refinement_level = self.params.get("refinement_level", 0)
                geo_path, _, _, out_stem = _paths_for_level(
                    refinement_level, out_folder="grids"
                )
                factory = GeoNestedRefinementFactory(
                    src_path=str(geo_path),
                    dim=dim,  # type:ignore
                    num_refinements=num_refinements,
                    out_stem=out_stem,
                )

                # Retrieve the coarse and the fine mdg. First item of the list
                # corresponds to the coarse mdg and second list correspond to fine mdg
                mdg_coarse = None
                mdg_fine = None
                for i, mdg in enumerate(factory):
                    if i == 0:
                        mdg_coarse = mdg
                    else:
                        mdg_fine = mdg

                # Sanity check
                if mdg_coarse is None or mdg_fine is None:
                    msg = "Nested refinement factory did not yield two levels."
                    raise RuntimeError(msg)
            else:
                raise ValueError("Unsupported refinement strategy.")

            # Prepare mapping dictionary to replace grids
            sd_map = {}

            # Get mapping of subdomains
            for dim in [2]:
                for sd_coarse, sd_fine in zip(
                    mdg_coarse.subdomains(dim=dim),  # type:ignore
                    mdg_fine.subdomains(dim=dim),  # type:ignore
                ):
                    assert sd_coarse.dim == sd_fine.dim
                    sd_map[sd_coarse] = sd_fine

            # Perform replacement
            mdg_coarse.replace_subdomains_and_interfaces(
                sd_map=sd_map,
                # interface_map=intf_map
            )

        # Finally, set mdg and fracture network as a public attribute
        self.fracture_network = fn
        self.mdg: pp.MixedDimensionalGrid = mdg_coarse  # type:ignore

        # Bookkeeping: dim, domain, fractures
        self.nd: int = self.mdg.dim_max()
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Update local projections
        pp.set_local_coordinate_projections(self.mdg)

        # Wells (unchanged)
        self.set_well_network()
        if len(self.well_network.wells) > 0:
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            self.well_network.mesh(self.mdg)
