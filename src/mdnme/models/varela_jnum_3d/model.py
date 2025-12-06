"""
This module contains a code verification implementation using a manufactured solution
for the three-dimensional, incompressible, single phase flow with a single,
fully embedded vertical fracture in the middle of the domain.

Details regarding the manufactured solution can be found in Appendix D.2 from [1].

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""

from __future__ import annotations

import porepy as pp

from mdnme.models.varela_jnum_2d.model import (
    VarelaJNumSetup2D,
    VarelaJNumSolutionStrategy2D,
)
from mdnme.models.varela_jnum_3d.exact_solution import VarelaJNumExactSolution3D
from mdnme.models.varela_jnum_3d.geometry import VarelaJNumGeometry3D

# PorePy typings
number = pp.number
grid = pp.GridLike


# -----> Solution strategy
class VarelaJNumSolutionStrategy3D(VarelaJNumSolutionStrategy2D):
    """Modified solution strategy for the verification model."""

    exact_sol: VarelaJNumExactSolution3D
    """Exact solution object."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: VarelaJNumExactSolution3D
        """Exact solution object."""

    def set_materials(self):
        """Set material constants for the verification model."""
        super().set_materials()
        # Instantiate exact solution object
        self.exact_sol = VarelaJNumExactSolution3D(self)


# -----> Mixer
class VarelaJNumSetup3D(  # type: ignore[misc]
    VarelaJNumGeometry3D,
    VarelaJNumSolutionStrategy3D,
    VarelaJNumSetup2D,
):
    """
    Mixer class for the 3d incompressible flow model with a single fracture.
    """