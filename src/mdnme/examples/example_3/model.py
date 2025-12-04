"""
This module contains the model for the Example 3 of the manuscript.
"""
import sys

from typing import Callable

import porepy as pp
import mdnme
import numpy as np

from mdnme.estimates.helpers import ErrorEstimatesSaveData
from mdnme.examples.example_3.boundary_conditions import (
    NoFluxBoundaryConditions,
    ModifiedBalanceEquation,
)
from mdnme.examples.example_3.geometry import GeometryNonMatching

from mdnme.estimates.error_estimation import (
    estimate_errors,
    compute_error_indicators,
    compute_local_errors,
    transfer_errors_iterate_solutions,
    aggregate_local_errors,
)

from porepy.examples.flow_benchmark_3d_case_3 import (
    FlowBenchmark3dCase3Model,
    FractureSolidConstants,
)

solid_constants = FractureSolidConstants(
    residual_aperture=1e-2,
    normal_permeability=1e4,
    fracture_permeability=1e4,
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

    def assign_ids_to_subdomains(self) -> None:
        """Assign ids to subdomains"""
        count = 0
        for sd, data in self.mdg.subdomains(dim=2, return_data=True):

            data[pp.ITERATE_SOLUTIONS]["ID"] = {}
            data[pp.ITERATE_SOLUTIONS]["ID"][0] = count * np.ones(sd.num_cells)

            data[pp.TIME_STEP_SOLUTIONS]["ID"] = {}
            data[pp.TIME_STEP_SOLUTIONS]["ID"][0] = count * np.ones(sd.num_cells)

            count += 1

    def visualize_fluid_sources(self) -> None:
        """Visualization of fluid sources."""
        for sd, data in self.mdg.subdomains(dim=2, return_data=True):

            sd_id_inj, cell_idx_inj = self._injector_idx(sd)
            sd_id_prd, cell_idx_prd = self._productor_idx(sd)

            val_loc = np.zeros(sd.num_cells)

            if sd.id == sd_id_inj:
                val_loc[cell_idx_inj] = -1
            if sd.id == sd_id_prd:
                val_loc[cell_idx_prd] = 1

            data[pp.ITERATE_SOLUTIONS]["well"] = {}
            data[pp.ITERATE_SOLUTIONS]["well"][0] = val_loc

            data[pp.TIME_STEP_SOLUTIONS]["well"] = {}
            data[pp.TIME_STEP_SOLUTIONS]["well"][0] = val_loc

    def get_list_of_sources(self) -> list[np.ndarray]:
        """Returns the all external sources imposed in the mdg."""

        def reshape(array: np.ndarray) -> np.ndarray:
            return np.reshape(array, (array.size, 1))

        sources = []
        for sd in self.mdg.subdomains():

            if sd.dim != 2:
                sources.append(reshape(np.zeros(sd.num_cells)))
            else:

                sd_id_inj, cell_idx_inj = self._injector_idx(sd)
                sd_id_prd, cell_idx_prd = self._productor_idx(sd)

                rate = self.params.get("source_rate", 1)
                source_loc = np.zeros(sd.num_cells)
                if sd.id == sd_id_inj:
                    source_loc[cell_idx_inj] = -rate / sd.cell_volumes[cell_idx_inj]

                if sd.id == sd_id_prd:
                    source_loc[cell_idx_prd] = rate / sd.cell_volumes[cell_idx_prd]

                sources.append(reshape(source_loc))

        return sources


    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""

        # Save error estimates data (needed for running the estimates machinery)
        self.error_estimates_data_saving()

        # Get list of source arrays
        source_list = self.get_list_of_sources()

        # Estimate errors
        is_non_matching = self.params.get("non_matching", False)
        estimate_errors(
            self.mdg,
            sources=source_list,
            is_non_matching=is_non_matching)

        # Compute local errors and error indicators
        compute_error_indicators(self.mdg)

        # Transfer from iterate to time step
        transfer_errors_iterate_solutions(self.mdg)

        # Print aggregated local errors
        if not is_non_matching:
            print('----- Matching Error Estimates ------')
        else:
            print('----- Non-matching Error Estimates ------')
        print(f"Majorant : {mdnme.estimates.error_estimation.get_majorant(self.mdg)}")
        local_errors = aggregate_local_errors(self.mdg)
        print(f"3D subdomain error: {local_errors['subdomain_error'][3]}")
        print(f"2D subdomain error: {local_errors['subdomain_error'][2]}")
        print(f"1D subdomain error: {local_errors['subdomain_error'][1]}")
        print(f"2D interface error: {local_errors['interface_error'][2]}")
        print(f"1D interface error: {local_errors['interface_error'][1]}")

        # Visualization methods
        self.assign_ids_to_subdomains()
        self.visualize_fluid_sources()

        # Export error indicators
        if self.params.get("export_to_vtu", False):
            self.exporter.write_vtu([
                "pressure",
                "ID",
                "well",
                "diffusive_error",
                "residual_error",
                "error_indicator",
                "interface_darcy_flux",
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
    ModifiedBalanceEquation,
    NoFluxBoundaryConditions,
    SmallFeaturesSolutionStrategy,
    FlowBenchmark3dCase3Model,
):
    """Main model for running the analysis corresponding to example number 3."""
