"""
This module contains the model for the Example 3 of the manuscript.
"""
import sys

from typing import Callable

import porepy as pp
import mdamr

from mdamr.estimates.helpers import ErrorEstimatesSaveData

from porepy.examples.flow_benchmark_3d_case_3 import (
    FlowBenchmark3dCase3Model,
    solid_constants,
)


class SmallFeaturesSolutionStrategy(
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
class SmallFeaturesModel(  # type: ignore[misc]
    ErrorEstimatesSaveData,
    SmallFeaturesSolutionStrategy,
    FlowBenchmark3dCase3Model,
):
    """Main model for running the analysis corresponding to example number 3."""


# # %% Runner
# params = {
#     "material_constants": {"solid": solid_constants},
#     "refinement_level": 0,
# }
# model = SmallFeaturesModel(params)
# pp.run_time_dependent_model(model, params)
