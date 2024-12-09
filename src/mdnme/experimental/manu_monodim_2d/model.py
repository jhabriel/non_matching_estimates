"""
This module contains a code verification implementation using a manufactured solution
for the two-dimensional, incompressible, single phase flow with a single, fully embedded
vertical fracture in the middle of the domain.

Details regarding the manufactured solution can be found in Appendix D.1 from [1].

References:

    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. M., & Radu, F. A.
      (2022). A posteriori error estimates for hierarchical mixed-dimensional
      elliptic equations. Journal of Numerical Mathematics.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import quadpy
import sympy as sym
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.utils.examples_utils import VerificationUtils
from porepy.viz.data_saving_model_mixin import VerificationDataSaving

import mdamr as amr

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
}


# -----> Data-saving
@dataclass
class ManuIncompSaveData:
    """Data class to save relevant results from the verification setup."""

    approx_flux: np.ndarray
    """Numerical flux."""

    approx_pressure: np.ndarray
    """Numerical pressure."""

    error_pressure: number
    """L2-discrete relative error for the pressure."""

    error_flux: number
    """L2-discrete relative error for the flux."""

    exact_pressure: np.ndarray
    """Exact pressure."""

    exact_flux: np.ndarray
    """Exact flux."""


class ManuIncompDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    exact_sol: Mono2dExactSolution
    """Exact solution object."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    def collect_data(self) -> ManuIncompSaveData:
        """Collect data from the verification setup.

        Returns:
            ManuIncompSaveData object containing the results of the verification.

        """

        sd: pp.Grid = self.mdg.subdomains()[0]
        exact_sol: Mono2dExactSolution = self.exact_sol

        # Collect data
        exact_pressure = exact_sol.pressure(sd)
        pressure_ad = self.pressure([sd])
        approx_pressure = pressure_ad.evaluate(self.equation_system).val
        error_pressure = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=exact_pressure,
            approx_array=approx_pressure,
            is_scalar=True,
            is_cc=True,
            relative=True,
        )

        exact_flux = exact_sol.flux(sd)
        flux_ad = self.darcy_flux([sd])
        approx_flux = flux_ad.evaluate(self.equation_system).val
        error_flux = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=exact_flux,
            approx_array=approx_flux,
            is_scalar=True,
            is_cc=False,
            relative=True,
        )

        # Store collected data in data class
        collected_data = ManuIncompSaveData(
            exact_pressure=exact_pressure,
            exact_flux=exact_flux,
            approx_pressure=approx_pressure,
            approx_flux=approx_flux,
            error_pressure=error_pressure,
            error_flux=error_flux,
        )

        return collected_data


# -----> Exact solution
class Mono2dExactSolution:
    """Class containing the exact manufactured solution for the verification setup."""

    def __init__(self, setup):
        """Constructor of the class."""

        # Symbolic variables
        x, y = sym.symbols("x y")

        # Exact pressure solution
        manufactured_sol = setup.params.get("manufactured_solution", "parabolic")
        if manufactured_sol == "parabolic":
            p = x * (1 - x) * y * (1 - y)
        elif manufactured_sol == "trigonometric":
            p = sym.sin(2 * sym.pi * x) * sym.cos(2 * sym.pi * y)
        else:
            raise NotImplementedError("Manufactured solution is not available.")

        # Exact Darcy flux
        q = [-sym.diff(p, x), -sym.diff(p, y)]

        # Exact source
        f = sym.diff(q[0], x) + sym.diff(q[1], y)

        # Public attributes
        self.p = p
        self.q = q
        self.f = f

    def pressure(self, sd: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix pressure at the cell centers.

        Parameters:
            sd: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Lambdify expression
        p_fun = sym.lambdify((x, y), self.p, "numpy")

        # Cell-centered pressures
        cc = sd.cell_centers
        p_cc = p_fun(cc[0], cc[1])

        return p_cc

    def flux(self, sd: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix Darcy flux at the face centers.

        Parameters:
            sd: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_matrix.face_normals``.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = sd.face_centers
        fn = sd.face_normals

        # Lambdify expression

        q_fun: list[Callable] = [
            sym.lambdify(
                (
                    x,
                    y,
                ),
                self.q[0],
                "numpy",
            ),
            sym.lambdify((x, y), self.q[1], "numpy"),
        ]

        # Face-centered Darcy fluxes
        q_fc = q_fun[0](fc[0], fc[1]) * fn[0] + q_fun[1](fc[0], fc[1]) * fn[1]

        return q_fc

    def source(self, sd: pp.Grid) -> np.ndarray:
        """Compute exact integrated matrix source.

        Parameters:
            sd: Grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Lambdify expression
        f_fun = sym.lambdify((x, y), self.f, "numpy")

        # Integrated cell-centered sources
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd)

        def integrand(x):
            return f_fun(x[0], x[1]) * np.ones_like(x[0])

        return int_method.integrate(integrand, elements)

    def boundary_values(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            boundary_grid: Matrix boundary grid.

        Returns:
            Array of ``shape=(boundary_grid_matrix.num_cells, )`` with the exact
            pressure values at the exterior boundary faces.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = boundary_grid.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify((x, y), self.p, "numpy")

        # Boundary pressures
        p_bf = p_fun(fc[0], fc[1])

        return p_bf

    def residual_error(self, sd: pp.Grid, d: dict) -> np.ndarray:
        """Compute square of residual errors for 2D (only the norm)"""
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Lambdify expression
        f_fun = sym.lambdify((x, y), self.f, "numpy")

        # Retrieve reconstructed velocity and compute divergence
        recon_u = d["estimates"]["recon_sd_flux"].copy()
        u = amr.utils.poly2col(recon_u)
        div_u = 2 * u[0]

        # Integration method and retrieving elements
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd)
        weights = (sd.cell_diameters() / np.pi) ** 2

        # Declare integrand
        def integrand(x):
            return (f_fun(x[0], x[1]) * np.ones_like(x[0]) - div_u) ** 2

        return weights * int_method.integrate(integrand, elements)


# -----> Utilities
class ManuIncompUtils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[ManuIncompSaveData]
    """List of ManuIncompSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_pressure()

    def _plot_pressure(self) -> None:
        """Plots exact and numerical pressures in the matrix."""
        sd = self.mdg.subdomains()[0]
        p_num = self.results[-1].approx_pressure
        p_ex = self.results[-1].exact_pressure
        pp.plot_grid(
            sd, p_ex, plot_2d=True, linewidth=0, title="Matrix pressure (Exact)"
        )
        pp.plot_grid(
            sd, p_num, plot_2d=True, linewidth=0, title="Matrix pressure (MPFA)"
        )


# -----> Geometry
class UnitSquareGrid:
    """Class for setting up the geometry of the unit square domain."""

    params: dict
    """Simulation model parameters."""

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(2, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        default_mesh_arguments = {"cell_size": 0.1}
        return self.params.get("meshing_arguments", default_mesh_arguments)


# -----> Boundary conditions
class ManuIncompBoundaryConditions(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    """Set boundary conditions for the simulation model."""

    exact_sol: Mono2dExactSolution
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
            vals[:] = self.exact_sol.boundary_values(boundary_grid=boundary_grid)
        return vals


# -----> Balance equations
class ManuIncompBalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    exact_sol: Mono2dExactSolution
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
        sd = self.mdg.subdomains()[0]
        external_sources = pp.wrap_as_ad_array(self.exact_sol.source(sd))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


# -----> Solution strategy
class ManuIncompSolutionStrategy2d(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    exact_sol: Mono2dExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    plot_results: Callable
    """Method to plot results of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    results: list[ManuIncompSaveData]
    """List of SaveData objects."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: Mono2dExactSolution
        """Exact solution object."""

        self.results: list[ManuIncompSaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks to guarantee the validity of the manufactured solution
        assert self.fluid.density() == 1
        assert self.fluid.viscosity() == 1
        assert self.fluid.compressibility() == 0
        assert self.solid.permeability() == 1

        # Instantiate exact solution object after materials have been set
        self.exact_sol = Mono2dExactSolution(self)

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
class Mono2d(  # type: ignore[misc]
    UnitSquareGrid,
    ManuIncompBalanceEquation,
    ManuIncompBoundaryConditions,
    ManuIncompSolutionStrategy2d,
    ManuIncompUtils,
    ManuIncompDataSaving,
    amr.ErrorEstimatesSaveData,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the 2d incompressible flow setup with a single fracture.
    """
