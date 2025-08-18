"""
This module contains a code verification implementation using a manufactured solution
for the two-dimensional, incompressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

Details regarding the manufactured solution can be found in Appendix D.1 from [1].

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. & Radu, F. (2023). A
      posteriori error estimates for hierarchical mixed-dimensional elliptic
      equations. Journal of Numerical Mathematics, 31(4), 247-280.
      https://doi.org/10.1515/jnma-2022-0038

"""
from __future__ import annotations

from typing import Callable

import numpy as np
import porepy as pp

import mdnme
from mdnme.examples.varela_jnum_2d.exact_solution import VarelaJNumExactSolution2d
from mdnme.examples.varela_jnum_2d.geometry import VarelaJNumGeometry2D

# Material constants for the verification setup. Constants with (**) cannot be
# changed since the manufactured solution implicitly assume such values.
manu_incomp_fluid: dict[str, pp.number] = {
    "compressibility": 0,  # (**)
    "density": 1.0,  # (**)
    "viscosity": 1.0,  # (**)
}

manu_incomp_solid: dict[str, pp.number] = {
    "residual_aperture": 1.0,  # (**)
    "permeability": 1.0,  # (**)
    "normal_permeability": 0.5,  # (**) counteracts division by a/2 in interface law
}


class VarelaJNumBoundaryConditions(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    """Set boundary conditions for the simulation model."""

    exact_sol: VarelaJNumExactSolution2d
    """Exact solution object."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition type."""
        if sd.dim == self.mdg.dim_max():  # Dirichlet for the matrix
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        else:  # Neumann for the fracture tips
            boundary_faces = self.domain_boundary_sides(sd).all_bf
            return pp.BoundaryCondition(sd, boundary_faces, "neu")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Analytical boundary condition values for Darcy flux.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        vals = np.zeros(boundary_grid.num_cells)
        if boundary_grid.dim == (self.mdg.dim_max() - 1):
            # Dirichlet for matrix
            vals[:] = self.exact_sol.boundary_values(boundary_grid_matrix=boundary_grid)
        return vals


class VarelaJNumBalanceEquation(pp.fluid_mass_balance.FluidMassBalanceEquations):
    """Modify balance equation to account for external sources."""

    exact_sol: VarelaJNumExactSolution2d
    """Exact solution object."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Contribution of mass fluid sources to the mass balance equation.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise Ad operator containing the fluid source contributions.

        """
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        values = []
        for sd in subdomains:
            if sd.dim == self.mdg.dim_max():
                values.append(self.exact_sol.integrated_matrix_source(sd_matrix=sd))
            else:
                values.append(self.exact_sol.integrated_fracture_source(sd_frac=sd))
        external_sources = pp.wrap_as_dense_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


class VarelaJNumSolutionStrategy2D(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    exact_sol: VarelaJNumExactSolution2d
    """Exact solution object."""

    fluid: pp.Fluid
    """Object containing the fluid constants."""

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: VarelaJNumExactSolution2d
        """Exact solution object."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks to guarantee the validity of the manufactured solution
        assert self.fluid.reference_component.density == 1
        assert self.fluid.reference_component.viscosity == 1
        assert self.fluid.reference_component.compressibility == 0
        assert self.solid.permeability == 1
        assert self.solid.residual_aperture == 1
        assert self.solid.normal_permeability == 0.5

        # Instantiate exact solution object after materials have been set
        self.exact_sol = VarelaJNumExactSolution2d()

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        # Save error estimates data
        self.error_estimates_data_saving()

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def _is_time_dependent(self) -> bool:
        """The problem is stationary."""
        return False


class VarelaJNumSetup2D(  # type: ignore[misc]
    VarelaJNumGeometry2D,
    VarelaJNumBalanceEquation,
    VarelaJNumBoundaryConditions,
    VarelaJNumSolutionStrategy2D,
    mdnme.ErrorEstimatesSaveData,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the 2d incompressible flow setup with a single fracture.
    """
