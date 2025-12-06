"""This module contains the model for the Example 2 of the manuscript."""

from typing import Callable

import numpy as np
import porepy as pp

from mdnme.estimates.error_estimation import (
    compute_error_indicators,
    estimate_errors,
    transfer_errors_iterate_solutions,
)
from mdnme.estimates.helpers import ErrorEstimatesSaveData
from mdnme.examples.example_2.boundary_conditions import BoundaryConditionsModified
from mdnme.examples.example_2.flow_benchmark_3d_case_2 import (
    solid_constants_conductive,  # to be imported by "example_2.model"
)
from mdnme.examples.example_2.flow_benchmark_3d_case_2 import FlowBenchmark3dCase2Model
from mdnme.examples.example_2.geometry import GeometryNonMatching


class Geiger3dSolutionStrategy(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    fluid: pp.Fluid
    """Object containing the fluid constants."""

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

    def save_inlet_and_outlet_cells(self):
        sd, data = self.mdg.subdomains(dim=3, return_data=True)[0]
        cc = sd.cell_centers

        outflow = np.logical_and.reduce(
            tuple(cc[i, :] > 0.875 + 1e-8 for i in range(3))
        )
        inflow = np.logical_and.reduce(
            tuple(cc[i, :] < 0.25 + 1e-8 for i in range(3))
        )

        bc_cells = np.zeros(sd.num_cells)
        bc_cells[outflow] = -1
        bc_cells[inflow] = 1

        data[pp.ITERATE_SOLUTIONS]['bc_cells'] = {}
        data[pp.ITERATE_SOLUTIONS]['bc_cells'][0] = bc_cells
        data[pp.TIME_STEP_SOLUTIONS]['bc_cells'] = {}
        data[pp.TIME_STEP_SOLUTIONS]['bc_cells'][0] = bc_cells

    def save_matrix_permeability_cells(self):
        sd, data = self.mdg.subdomains(dim=3, return_data=True)[0]

        low_perm = self._low_perm_zones(sd)
        low_perm_cells = np.zeros(sd.num_cells)
        low_perm_cells[low_perm] = 1

        data[pp.ITERATE_SOLUTIONS]['low_perm_cells'] = {}
        data[pp.ITERATE_SOLUTIONS]['low_perm_cells'][0] = low_perm_cells

        data[pp.TIME_STEP_SOLUTIONS]['low_perm_cells'] = {}
        data[pp.TIME_STEP_SOLUTIONS]['low_perm_cells'][0] = low_perm_cells

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        # Save error estimates data
        self.error_estimates_data_saving()

        # Retrieve zones for visualization
        self.save_inlet_and_outlet_cells()
        self.save_matrix_permeability_cells()

        # Estimate errors
        is_non_matching = self.params.get("non_matching", False)
        estimate_errors(self.mdg, is_non_matching=is_non_matching)

        # Compute error indicators
        compute_error_indicators(self.mdg)

        # Transfer from iterate to time step
        transfer_errors_iterate_solutions(self.mdg)

        # Export error indicators
        if self.params.get("export_results", False):
            self.exporter.write_vtu([
                "pressure",
                "interface_darcy_flux",
                "diffusive_error",
                "residual_error",
                "error_indicator",
                "bc_cells",
                "low_perm_cells",
            ])
            
    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def _is_time_dependent(self) -> bool:
        """The problem is stationary."""
        return False


# %% Mixer
class Geiger3dModel(  # type: ignore[misc]
    GeometryNonMatching,
    ErrorEstimatesSaveData,
    Geiger3dSolutionStrategy,
    BoundaryConditionsModified,
    FlowBenchmark3dCase2Model,
):
    """Main model for running the analysis corresponding to example number 2."""
