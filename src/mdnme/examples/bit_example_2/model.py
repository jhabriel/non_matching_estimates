"""
This module contains the model for the Example 2 of the manuscript.
"""
import sys

from typing import Callable

import porepy as pp
import mdnme

from mdnme.estimates.helpers import ErrorEstimatesSaveData
from mdnme.examples.bit_example_2.boundary_conditions import BoundaryConditionsModified
from mdnme.examples.bit_example_2.geometry import GeometryNonMatching
from mdnme.examples.bit_example_2.flow_benchmark_3d_case_2 import (
    FlowBenchmark3dCase2Model,
    solid_constants_conductive,
    solid_constants_blocking,
)

from mdnme.estimates.error_estimation import (
    estimate_errors,
    compute_error_indicators,
    transfer_errors_iterate_solutions,
    compute_sd_and_intf_errors_of_equal_dim,
)



class SmallFeaturesSolutionStrategy(
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

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        # Save error estimates data
        self.error_estimates_data_saving()

        # Estimate errors
        estimate_errors(self.mdg)

        # Compute error indicators
        compute_error_indicators(self.mdg)

        # Transfer from iterate to time step
        transfer_errors_iterate_solutions(self.mdg)

        # Export error indicators
        if self.params.get("export_to_vtu", False):
            self.exporter.write_vtu([
                "pressure",
                "diffusive_error",
                "residual_error",
                "error_indicator",
            ])

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def _is_time_dependent(self) -> bool:
        """The problem is stationary."""
        return False


# %% Mixer
class SmallFeaturesModel(  # type: ignore[misc]
    GeometryNonMatching,
    ErrorEstimatesSaveData,
    SmallFeaturesSolutionStrategy,
    BoundaryConditionsModified,
    FlowBenchmark3dCase3Model,
):
    """Main model for running the analysis corresponding to example number 3."""


# # %% Runner
# params = {
#     "material_constants": {"solid": solid_constants},
#     "refinement_level": 0,
#     "non_matching": True,
# }
# model = SmallFeaturesModel(params)
# pp.run_time_dependent_model(model, params)
#
