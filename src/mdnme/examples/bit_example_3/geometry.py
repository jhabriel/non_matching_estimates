"""Module containing geometry-related mixins for the third numerical example."""

import porepy as pp
import numpy as np

from typing import cast

from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from mdnme.utils.nested_refinement import GeoNestedRefinementFactory
from mdnme.utils.grid_rotation import assign_canonical_rotations

class GeometryNonMatching(pp.PorePyModel):
    """Define Geometry as specified in Section 5.3 of the benchmark study [1]."""

    def set_geometry(self) -> None:
        """Create mixed-dimensional grid and fracture network."""

        # Create mixed-dimensional grid and fracture network.
        self.mdg, self.fracture_network = benchmark_3d_case_3(
            refinement_level=self.params.get("refinement_level", 0)
        )
        self.nd: int = self.mdg.dim_max()

        # Obtain domain and fracture list directly from the fracture network.
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        if self.params.get("non_matching", False):
            # Do global refinement and replace all lower-dimensional subdomain grids
            # to generate non-matching grids
            ref_lev = self.params.get("refinement_level", 0)
            if ref_lev == 0:
                src_path = "mesh30k.geo"
            elif ref_lev == 1:
                src_path = "mesh140k.geo"
            elif ref_lev == 2:
                src_path = "mesh350k.geo"
            elif ref_lev == 3:
                src_path = "mesh500k.geo"
            else:
                raise ValueError("Expected refinement level 0, 1, 2, or 3.")

            dim = 3
            num_refinements = 1
            out_stem= f'{ref_lev}_nonmatch'

            factory = GeoNestedRefinementFactory(
                src_path, dim, num_refinements, out_stem
            )

            # Retrieve the mixed-dimensional grids
            for item, mdg in enumerate(factory):
                if item == 0:
                    mdg_coarse = mdg
                else:
                    mdg_fine = mdg

            # Create map from coarse to fine subdomains
            sd_map = {}
            for sd_co, sd_fi in zip(mdg_coarse.subdomains(), mdg_fine.subdomains()):
                dim_coarse = sd_co.dim
                dim_fine = sd_fi.dim
                assert dim_coarse == dim_fine
                sd_dim = dim_coarse
                if sd_dim < 3:
                    sd_map[sd_co] = sd_fi

            mdg_coarse.replace_subdomains_and_interfaces(sd_map=sd_map)

        # Overwrite previous mixed-dimensional grid if non_matching = True
        # Also, make sure to recompute local projections
        if self.params.get("non_matching", False):
            self.mdg = mdg_coarse
            pp.set_local_coordinate_projections(self.mdg)
            assign_canonical_rotations(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections.
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)