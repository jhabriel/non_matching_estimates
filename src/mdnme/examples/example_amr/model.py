# Create a model with an L-shaped domain


import porepy as pp
import numpy as np
import mdnme


from porepy.models.fluid_mass_balance import (
    SinglePhaseFlow,
    BoundaryConditionsSinglePhaseFlow,
)

from mdnme.estimates.error_estimation import estimate_errors, compute_error_indicators

from typing import Callable

from mdnme.porepy_interface.fem_grid_to_porepy_grid import fem_grid_to_sd_grid_2d
from mdnme.porepy_interface.porepy_grid_to_fem_grid import porepy_grid_to_fem_grid
from mdnme.amr.refinement_utils import plot_fem_mesh

class LShapedGeometry:

    def set_geometry(self) -> None:

        # Create the geometry through domain and fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(
            self.fractures,
            self.domain
        )
        # If the AMR is off, we produce the mdg in the usual way, otherwise we pass
        # the mdg (obtained by replacing the refined grids) in the old model through
        # the params dictionary
        amr = self.params.get("amr", "off")
        if amr == "off":
            # Create a mixed-dimensional grid.
            self.mdg = pp.create_mdg(
                self.grid_type(),
                self.meshing_arguments(),
                self.fracture_network,
                **self.meshing_kwargs(),
            )
        else:
            # Mixed-dimensional grid is provided
            self.mdg = self.params["mdg"]
            # We have to make sure that the boundary grids have their geometry computed
            for sd in self.mdg.subdomains():
                bg = self.mdg.subdomain_to_boundary_grid(sd)
                bg.compute_geometry()

        # Dimensionality of highest-dimensional manifold
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

    def set_domain(self) -> None:
        """Define an L-shaped domain using the polytope declaration."""
        # Declare domain from polygon
        line_1 = np.array([[0, 2], [0, 0]])
        line_2 = np.array([[2, 2], [0, 1]])
        line_3 = np.array([[2, 1], [1, 1]])
        line_4 = np.array([[1, 1], [1, 2]])
        line_5 = np.array([[1, 0], [2, 2]])
        line_6 = np.array([[0, 0], [2, 0]])

        domain = pp.Domain(polytope=[line_1, line_2, line_3, line_4, line_5, line_6])
        self._domain = domain

    def grid_type(self) -> str:
        """Use simplex by default"""
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Use 1.0 as the cell size."""
        return self.params.get("meshing_arguments", {"cell_size": 1.0})


class LShapedBoundaryConditions(BoundaryConditionsSinglePhaseFlow):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.west + bounds.east, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 1 on west side, 0 on east side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        # See section on scaling for explanation of the conversion.
        values[bounds.west] = self.fluid.convert_units(1, "Pa")
        values[bounds.east] = self.fluid.convert_units(0, "Pa")
        return values


class LShapedSolutionStrategy(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

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


# Mixer class
class LShapedModel(
    LShapedGeometry,
    LShapedBoundaryConditions,
    LShapedSolutionStrategy,
    mdnme.ErrorEstimatesSaveData,
    SinglePhaseFlow,
):
    """L-shaped domain model."""


# %% Run model

# Define material constants
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

solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}

# Run model
params = {
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.25},
}

tol = 0.01
max_refinement_steps = 10
refinement_step = 0

# Main AMR loop
while refinement_step < max_refinement_steps:

    # SOLVE
    if refinement_step == 0:
        model = LShapedModel(params)
        pp.run_time_dependent_model(model, {})
    else:
        params["amr"] = "on"
        params["prepare_simulation"] = False
        params["mdg"] = model.mdg
        model = LShapedModel(params)
        pp.run_time_dependent_model(model, {})

    # ESTIMATE
    estimate_errors(model.mdg, "keilegavlen_p1")
    compute_error_indicators(model.mdg)
    g, d = model.mdg.subdomains(return_data=True)[0]
    indicators = d["estimates"]["error_indicator"]

    # MARK
    min_error = np.min(indicators)
    max_error = np.max(indicators)
    if refinement_step == 0:
        color_map_values = [min_error, max_error]

    if max_error < tol:
        print(f"Stopping refinement: max error {max_error} < tolerance {tol}.")
        break

    marked_elements = mdnme.doerfler_marking(indicators, theta=0.6)

    # threshold = indicators > 0.8 * np.max(indicators)
    # marked_elements = np.zeros(g.num_cells, dtype=bool)
    # marked_elements[threshold] = indicators[threshold]

    # REFINE
    coordinates, elements = porepy_grid_to_fem_grid(g)

    # Use RGB
    new_coordinates, new_elements, *_ = mdnme.refine_rgb(
        coordinates=coordinates,
        elements=elements,
        marked_elements=marked_elements,
    )

    new_grid = fem_grid_to_sd_grid_2d(new_coordinates, new_elements)
    model.mdg.replace_subdomains_and_interfaces(sd_map={g: new_grid})
    # pp.save_img(f"indicator{refinement_step}", indicators, plot_2d=True)

    # PLOT GRID
    print(f"Refinement step {refinement_step}: mesh refined, max error {max_error}")
    pp.save_img(
        name=f"indicators{refinement_step}.png",
        grid=g,
        cell_value=indicators,
        plot_2d=True,
        title=f"Refinement Step {refinement_step}",
        **{"color_map_limits": color_map_values},
    )

    # INCREMENT COUNTER
    refinement_step += 1
