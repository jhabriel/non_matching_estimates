"""
This module contains the model for the Example 2 of the manuscript.
"""
from typing import Callable

import porepy as pp
import mdnme

from porepy.examples.flow_benchmark_2d_case_3 import (
    FlowBenchmark2dCase3bModel,
    solid_constants,
)
from mdnme.estimates.helpers import ErrorEstimatesSaveData


class ComplexNetworkSolutionStrategy(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    fluid: pp.FluidComponent
    """Object containing the fluid constants."""

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""
        super().__init__(params)

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


# %% Mixer
class ComplexNetworkModel(  # type: ignore[misc]
    ErrorEstimatesSaveData,
    ComplexNetworkSolutionStrategy,
    FlowBenchmark2dCase3bModel,
):
    """Model for example 2 of the manuscript.

    Example:

        params = {
                "material_constants": {"solid": solid_constants},
                "grid_type": "simplex",
                "meshing_arguments": {"cell_size": 0.1},
        }
        model = ComplexNetworkModel(params)
        pp.run_time_dependent_model(model, params)
        title = f"Pressure distribution. \n Flow in x-direction."
        pp.plot_grid(
            model.mdg,
            model.pressure_variable,
            figsize=(12, 10),
            plot_2d=True,
            title=title,
            pointsize=20,
            fracturewidth_1d=3,
            linewidth=0.5
        )

    """
