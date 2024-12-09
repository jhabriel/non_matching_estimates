from typing import Callable, Literal

import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.fracs.fracture_network import FractureNetwork3d


class TwoFullyEmbeddedCrossingFractures:
    """Generate fracture network and mixed-dimensional grid."""

    # Define attributes to be assigned later
    fracture_network: pp.fracture_network
    """Representation of fracture network including intersections."""
    well_network: pp.WellNetwork3d
    """Well network."""
    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid. Set by the method :meth:`set_md_grid`."""
    nd: int
    """Ambient dimension of the problem. Set by the method :meth:`set_geometry`"""
    units: pp.Units
    """Unit system."""
    params: dict
    """Parameters for the model."""
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    def set_geometry(self) -> None:
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)
        # Create a mixed-dimensional grid.
        self.mdg = pp.create_mdg(
            self.grid_type(),
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        # Assign tags to one-dimensional subdomains
        self.identify_1d_subdomains()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

    def identify_1d_subdomains(self) -> None:
        """

        :return:
        """
        for sd, d in self.mdg.subdomains(return_data=True, dim=1):
            cc = sd.cell_centers
            if np.all(cc[0] == 0.5):
                # This is a vertical fracture
                if np.mean(cc[1]) > 0.5:
                    sd.label = "north"
                else:
                    sd.label = "south"
            else:
                # This is a horizontal fracture
                if np.mean(cc[0]) > 0.5:
                    sd.label = "east"
                else:
                    sd.label = "west"

    def set_fractures(self) -> None:
        """Declare set of fractures.

        Note:
            For simplicial grids, several ghost fractures are needed to conform to the
            different subregions.

        """
        f1 = pp.LineFracture(np.array([[0.25, 0.50], [0.50, 0.50]]))
        f2 = pp.LineFracture(np.array([[0.50, 0.75], [0.50, 0.50]]))
        f3 = pp.LineFracture(np.array([[0.50, 0.50], [0.25, 0.50]]))
        f4 = pp.LineFracture(np.array([[0.50, 0.50], [0.50, 0.75]]))

        if self.grid_type() == "simplex":
            f1_ghost = pp.LineFracture(np.array([[0.00, 1.00], [0.75, 0.75]]))
            f2_ghost = pp.LineFracture(np.array([[0.00, 1.00], [0.25, 0.25]]))
            f3_ghost = pp.LineFracture(np.array([[0.25, 0.25], [0.00, 1.00]]))
            f4_ghost = pp.LineFracture(np.array([[0.75, 0.75], [0.00, 1.00]]))
            f5_ghost = pp.LineFracture(np.array([[0.25, 0.00], [0.50, 0.50]]))
            f6_ghost = pp.LineFracture(np.array([[0.75, 1.00], [0.50, 0.50]]))
            f7_ghost = pp.LineFracture(np.array([[0.50, 0.50], [0.75, 1.00]]))
            f8_ghost = pp.LineFracture(np.array([[0.50, 0.50], [0.25, 0.00]]))

            self._fractures = [
                f1,
                f2,
                f3,
                f4,
                f1_ghost,
                f2_ghost,
                f3_ghost,
                f4_ghost,
                f5_ghost,
                f6_ghost,
                f7_ghost,
                f8_ghost,
            ]
        elif self.grid_type() == "cartesian":
            self._fractures = [f1, f2, f3, f4]
        else:
            raise NotImplementedError()

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(dimension=2, size=1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Define mesh arguments for meshing."""
        return self.params.get("meshing_arguments", {"cell_size": 0.125})

    def meshing_kwargs(self) -> dict:
        """Declare meshing constraints. Ignore ghost fractures."""
        kw_args = {}
        if self.grid_type() == "simplex":
            kw_args = {"constraints": np.array([4, 5, 6, 7, 8, 9, 10, 11])}
        return kw_args
