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
from porepy.utils.default_domains import UnitSquareDomain


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
        self._domain = pp.Domain(UnitSquareDomain())

    def meshing_arguments(self) -> dict[str, float]:
        """Define mesh arguments for meshing."""
        return self.params.get("meshing_arguments", {"cell_size": 0.125})

    def meshing_kwargs(self) -> dict:
        """Declare meshing constraints. Ignore fractures 1 and 2."""
        kw_args = {}
        if self.grid_type() == "simplex":
            kw_args = {"constraints": np.array([1, 2])}
        return kw_args
