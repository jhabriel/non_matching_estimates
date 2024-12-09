from typing import Callable, Literal

import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain


class TwoFullyEmbeddedCrossingFractures:
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters."""

    grid_type: Callable[[], Literal["cartesian", "simplex", "tensor_grid"]]
    """Type of grid."""

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
            f1_ghost = pp.LineFracture(np.array([[0.25, 0.75], [0.75, 0.75]]))
            f2_ghost = pp.LineFracture(np.array([[0.75, 0.75], [0.75, 0.25]]))
            f3_ghost = pp.LineFracture(np.array([[0.75, 0.25], [0.25, 0.25]]))
            f4_ghost = pp.LineFracture(np.array([[0.25, 0.25], [0.25, 0.75]]))
            f5_ghost = pp.LineFracture(np.array([[0.00, 0.25], [1.00, 0.75]]))
            f6_ghost = pp.LineFracture(np.array([[1.00, 0.75], [1.00, 0.75]]))
            f7_ghost = pp.LineFracture(np.array([[1.00, 0.75], [0.00, 0.25]]))
            f8_ghost = pp.LineFracture(np.array([[0.00, 0.25], [0.00, 0.25]]))
            f9_ghost = pp.LineFracture(np.array([[0.25, 0.50], [0.75, 0.50]]))
            f10_ghost = pp.LineFracture(np.array([[0.75, 0.50], [0.75, 0.50]]))
            f11_ghost = pp.LineFracture(np.array([[0.75, 0.50], [0.25, 0.50]]))
            f12_ghost = pp.LineFracture(np.array([[0.25, 0.50], [0.25, 0.50]]))
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
                f9_ghost,
                f10_ghost,
                f11_ghost,
                f12_ghost,
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
        """Declare meshing constraints. Ignore fractures 1 and 2."""
        kw_args = {}
        if self.grid_type() == "simplex":
            kw_args = {
                "constraints": np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            }
        return kw_args
