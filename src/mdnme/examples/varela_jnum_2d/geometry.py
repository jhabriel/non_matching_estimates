"""
Module containing the mixin class to generate the mixed-dimensional grid associated
to the geometry used in the manufactured solution from Appendix D.1. from [1].

Reference:
    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. & Radu, F. (2023). A
      posteriori error estimates for hierarchical mixed-dimensional elliptic
      equations. Journal of Numerical Mathematics, 31(4), 247-280.
      https://doi.org/10.1515/jnma-2022-0038

"""


from typing import Callable, Literal

import numpy as np
import porepy as pp


class Varela2023JNumGeometry:
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters."""

    grid_type: Callable[[], Literal["cartesian", "simplex", "tensor_grid"]]
    """Type of grid."""

    def set_fractures(self) -> None:
        """Declare set of fractures.

        Note:
            For simplicial grids, two horizontal fractures at :math:`y = 0.25` and
            :math:`y = 0.75` are included in the fracture network to force the grid
            to conform to certain regions of the domain. Note, however, that these
            fractures will not be part of the mixed-dimensional grid.

        """
        physical_frac_0 = pp.LineFracture(np.array([[0.50, 0.50], [0.25, 0.75]]))

        if self.grid_type() == "simplex":
            ghost_frac_0 = pp.LineFracture(np.array([[0.00, 1.00], [0.25, 0.25]]))
            ghost_frac_1 = pp.LineFracture(np.array([[0.00, 1.00], [0.75, 0.75]]))
            self._fractures = [physical_frac_0, ghost_frac_0, ghost_frac_1]
        elif self.grid_type() == "cartesian":
            self._fractures = [physical_frac_0]
        else:
            raise NotImplementedError()

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

    def meshing_arguments(self) -> dict[str, float]:
        """Define mesh arguments for meshing."""
        return self.params.get("meshing_arguments", {"cell_size": 0.125})

    def meshing_kwargs(self) -> dict:
        """Declare meshing constraints. Ignore fractures 1 and 2."""
        kw_args = {}
        if self.grid_type() == "simplex":
            kw_args = {"constraints": np.array([1, 2])}
        return kw_args

    def set_geometry(self) -> None:

        # Create the geometry through domain and fracture set.
        self.set_domain()
        self.set_fractures()

        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(
            self.fractures,
            self.domain
        )

        # Generate the mixed-dimensional grid

        # If `non_matching_cell_sizes` is not given, then the mdg is matching, and
        # we create it in the usual way
        if (
                self.params.get("non_matching_cell_sizes") is None and
                self.params.get("full_non_matching_cell_sizes") is None
            ):

            mdg_final = pp.create_mdg(
                self.grid_type(),
                self.meshing_arguments(),
                self.fracture_network,
                **self.meshing_kwargs(),
            )

        # If `non_matching_cell_sizes` is given, we create the non-matching mdg
        elif self.params.get("non_matching_cell_sizes") is not None:

            # Retrieve the non-matching cell sizes. This is 3-tuple of floats. The
            # first element is the target cell size for the matrix, the second
            # element is the target cell size for the interface grid, and the third
            # element is the target cell size for the fracture grid
            cell_sizes: tuple[float, float, float] = self.params[
                "non_matching_cell_sizes"
            ]
            assert len(cell_sizes) == 3

            # The idea is to create three different fully matching mdgs and then
            # retrieve the fracture and interface grids from these mdgs and replace
            # into the mdg containing the "correct" higher-dimensional subdomain
            # This is quite a lazy solution, but it works.
            mdgs: list[pp.MixedDimensionalGrid] = []
            for cell_size in cell_sizes:
                mdg = pp.create_mdg(
                    self.grid_type(),
                    {"cell_size": cell_size},
                    self.fracture_network,
                    **self.meshing_kwargs(),
                )
                mdgs.append(mdg)

            # Now we replace the grids and produce the non-matching mixed-dimensional
            # grid. We keep the matrix fixed, and replace the fractures and mortar grids
            mdg_final = mdgs[0]
            mdg_final.replace_subdomains_and_interfaces(
                sd_map={mdg_final.subdomains()[1]: mdgs[2].subdomains()[1]},
                interface_map={mdg_final.interfaces()[0]: mdgs[1].interfaces()[0]},
            )

            # Make sure the geometry of the replaced boundary grids are computed.
            # TODO: This has to be taken care by PorePy.
            for sd in mdg_final.subdomains():
                bg = mdg_final.subdomain_to_boundary_grid(sd)
                bg.compute_geometry()

        else:

            # Retrieve the non-matching cell sizes. This is 3-tuple of floats. The
            # first element is the target cell size for the matrix, the second
            # element is the target cell size for the interface grid, and the third
            # element is the target cell size for the fracture grid
            cell_sizes: tuple[float, float, float, float, float] = self.params[
                "full_non_matching_cell_sizes"
            ]
            assert len(cell_sizes) == 5

            # Make sure the matrix size is of the same size at both sides of the
            # interface
            assert cell_sizes[0] == cell_sizes[-1]

            # The idea is to create four different fully matching mdgs and then
            # retrieve the fracture and interface side grids from these mdgs and replace
            # into the mdg containing the "correct" higher-dimensional subdomain
            # This is quite a lazy solution, but it works.
            mdgs: list[pp.MixedDimensionalGrid] = []
            for cell_size in cell_sizes:
                mdg = pp.create_mdg(
                    self.grid_type(),
                    {"cell_size": cell_size},
                    self.fracture_network,
                    **self.meshing_kwargs(),
                )
                mdgs.append(mdg)

            # Retrieve grids to be replaced
            # 0 : matrix grid
            # 1 : left interface grid
            # 2 : fracture grid
            # 3 : right interface grid
            # 4 : matrix grid

            # Get left interface side grid
            left_intf = mdgs[1].interfaces()[0]
            left_mortar_side =  left_intf.sides[0]
            left_side_grid = left_intf.side_grids[left_mortar_side]

            # Get fracture grid
            frac_grid = mdgs[2].subdomains()[1]

            # Get right interface side grid
            right_intf = mdgs[3].interfaces()[0]
            right_mortar_side = right_intf.sides[1]
            right_side_grid = right_intf.side_grids[right_mortar_side]

            # Now we replace the grids and produce the non-matching mixed-dimensional
            # grid. We keep the matrix fixed, and replace the fractures and mortar grids
            mdg_final = mdgs[0]
            mdg_final.replace_subdomains_and_interfaces(
                sd_map={mdg_final.subdomains()[1]: frac_grid},
                interface_map={mdg_final.interfaces()[0]:
                              {
                                  left_mortar_side: left_side_grid,
                                  right_mortar_side: right_side_grid,
                              }
                         }
                ,
            )

            # Make sure the geometry of the replaced boundary grids are computed.
            # TODO: This has to be taken care by PorePy.
            for sd in mdg_final.subdomains():
                bg = mdg_final.subdomain_to_boundary_grid(sd)
                bg.compute_geometry()

        # Finally, we have our mdg
        self.mdg = mdg_final

        # Dimensionality of highest-dimensional manifold
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)
