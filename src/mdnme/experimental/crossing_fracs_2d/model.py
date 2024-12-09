"""
Two-crossing fractures with manufactured solution model.

"""
from __future__ import annotations

from typing import Callable

import numpy as np
import porepy as pp
from data_saving import TwoCrossingDataSaving, TwoCrossingSaveData
from exact_sol import TwoCrossingExactSolution
from geometry import TwoFullyEmbeddedCrossingFractures
from porepy.utils.examples_utils import VerificationUtils

import mdnme as amr

# PorePy typings
number = pp.number
grid = pp.GridLike

# Material constants for the verification setup. Constants with (**) cannot be
# changed since the manufactured solution implicitly assume such values.
manu_incomp_fluid: dict[str, number] = {
    "compressibility": 0,  # (**)
    "density": 1.0,  # (**)
    "viscosity": 1.0,  # (**)
}

manu_incomp_solid: dict[str, number] = {
    "residual_aperture": 1.0,  # (**)
    "permeability": 1.0,  # (**)
    "normal_permeability": 0.5,  # (**) counteracts division by a/2 in interface law
}


# -----> Utilities
class ManuIncompUtils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[TwoCrossingSaveData]
    """List of ManuIncompSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_matrix_pressure()
        # self._plot_fracture_pressure()
        # self._plot_interface_fluxes()
        # self._plot_fracture_fluxes()

    def _plot_matrix_pressure(self) -> None:
        """Plots exact and numerical pressures in the matrix."""
        sd_matrix = self.mdg.subdomains()[0]
        p_num = self.results[-1].approx_p_2d
        p_ex = self.results[-1].exact_p_2d
        pp.plot_grid(
            sd_matrix, p_ex, plot_2d=True, linewidth=0, title="Matrix pressure (Exact)"
        )
        pp.plot_grid(
            sd_matrix, p_num, plot_2d=True, linewidth=0, title="Matrix pressure (MPFA)"
        )

    # def _plot_fracture_pressure(self):
    #     """Plots exact and numerical pressures in the fracture."""
    #     sd_frac = self.mdg.subdomains()[1]
    #     cc = sd_frac.cell_centers
    #     p_num = self.results[-1].approx_frac_pressure
    #     p_ex = self.results[-1].exact_frac_pressure
    #     plt.plot(p_ex, cc[1], label="Exact", linewidth=3, alpha=0.5)
    #     plt.plot(p_num, cc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
    #     plt.xlabel("Fracture pressure")
    #     plt.ylabel("y-coordinate")
    #     plt.legend()
    #     plt.show()
    #
    # def _plot_interface_fluxes(self):
    #     """Plots exact and numerical interface fluxes."""
    #     intf = self.mdg.interfaces()[0]
    #     cc = intf.cell_centers
    #     lmbda_num = self.results[-1].approx_intf_flux
    #     lmbda_ex = self.results[-1].exact_intf_flux
    #     plt.plot(lmbda_ex, cc[1], label="Exact", linewidth=3, alpha=0.5)
    #     plt.plot(lmbda_num, cc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
    #     plt.xlabel("Interface flux")
    #     plt.ylabel("y-coordinate")
    #     plt.legend()
    #     plt.show()
    #
    # def _plot_fracture_fluxes(self):
    #     """Plots exact and numerical fracture fluxes."""
    #     sd_frac = self.mdg.subdomains()[1]
    #     fc = sd_frac.face_centers
    #     q_num = self.results[-1].approx_frac_flux
    #     q_ex = self.results[-1].exact_frac_flux
    #     plt.plot(q_ex, fc[1], label="Exact", linewidth=3, alpha=0.5)
    #     plt.plot(q_num, fc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
    #     plt.xlabel("Fracture Darcy flux")
    #     plt.ylabel("y-coordinate")
    #     plt.legend()
    #     plt.show()


# -----> Boundary conditions
class TwoCrossingBoundaryConditions(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    """Set boundary conditions for the simulation model."""

    exact_sol: TwoCrossingExactSolution
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
            vals[:] = self.exact_sol.boundary_values(bg_2d=boundary_grid)
        return vals


# -----> Balance equations
class TwoCrossingBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    exact_sol: TwoCrossingExactSolution
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
            if sd.id == 0:
                values.append(self.exact_sol.f_2d(sd))
            elif sd.id == 1:
                values.append(self.exact_sol.f_1d_west(sd))
            elif sd.id == 2:
                values.append(self.exact_sol.f_1d_east(sd))
            elif sd.id == 3:
                values.append(self.exact_sol.f_1d_south(sd))
            elif sd.id == 4:
                values.append(self.exact_sol.f_1d_north(sd))
            elif sd.id == 5:
                values.append(self.exact_sol.f_0d(sd))
            else:
                raise ValueError()

        external_sources = pp.wrap_as_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


# -----> Solution strategy
class TwoCrossingSolutionStrategy(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    exact_sol: TwoCrossingExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    plot_results: Callable
    """Method to plot results of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    results: list[TwoCrossingSaveData]
    """List of SaveData objects."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: TwoCrossingExactSolution
        """Exact solution object."""

        self.results: list[TwoCrossingSaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks to guarantee the validity of the manufactured solution
        assert self.fluid.density() == 1
        assert self.fluid.viscosity() == 1
        assert self.fluid.compressibility() == 0
        assert self.solid.permeability() == 1
        assert self.solid.residual_aperture() == 1
        assert self.solid.normal_permeability() == 0.5

        # Instantiate exact solution object after materials have been set
        self.exact_sol = TwoCrossingExactSolution()

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()
        # Save error estimates data
        self.error_estimates_data_saving()

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def _is_time_dependent(self) -> bool:
        """The problem is stationary."""
        return False


# -----> Mixer
class TwoCrossingSetup(  # type: ignore[misc]
    TwoFullyEmbeddedCrossingFractures,
    TwoCrossingBalanceEquation,
    TwoCrossingBoundaryConditions,
    TwoCrossingSolutionStrategy,
    ManuIncompUtils,
    TwoCrossingDataSaving,
    amr.ErrorEstimatesSaveData,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the 2d incompressible flow setup with two crossing fractures.
    """
