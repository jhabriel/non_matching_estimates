from typing import Callable

import numpy as np
import porepy as pp
import pytest
from porepy.fracs.fracture_network_3d import FractureNetwork3d

from mdnme.estimates.error_estimation import (
    compute_error_indicators,
    estimate_errors,
    transfer_errors_iterate_solutions,
)
from mdnme.estimates.helpers import ErrorEstimatesSaveData


class Geometry(pp.PorePyModel):
    """Generate fracture network and mixed-dimensional grid."""

    def set_fractures(self) -> None:
        """Declare set of fractures.

        Note:
            The physical fracture is `physical_frac_0`. For simplices, fractures
            `ghost_frac_0` ... `ghost_frac_23` are constraints for the meshing process.

        """

        points_f1 = np.array(
            [
                [0.25, 0.75],
                [0.25, 0.75],
            ]
        )
        points_f2 = np.array(
            [
                [0.25, 0.75],
                [0.75, 0.25],
            ]
        )

        f1 = pp.LineFracture(points_f1)
        f2 = pp.LineFracture(points_f2)
        self._fractures = [f1, f2]

    def set_domain(self) -> None:
        """Set domain"""

        self._domain = pp.Domain(
            {
                "xmin": 0.0,
                "xmax": 1.0,
                "ymin": 0.0,
                "ymax": 1.0,
            }
        )

    def meshing_arguments(self) -> dict[str, float]:
        """Define mesh arguments for meshing."""
        return self.params.get("meshing_arguments", {"cell_size": 0.10})

    def set_geometry(self) -> None:

        # Create the geometry through domain and fracture set.
        self.set_domain()
        self.set_fractures()

        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(
            self._fractures,
            self._domain,
        )

        # Produce mdg
        if self.params.get("non_matching", False):

            # Create a matching mdg first
            mdg_coarse = pp.create_mdg(
                grid_type=self.grid_type(),
                meshing_args=self.meshing_arguments(),
                fracture_network=self.fracture_network,
            )

            # Retrieve target mesh size
            h = self.params["meshing_arguments"]["cell_size"]

            # Create a matching grid with half the refinement
            mdg_fine = pp.create_mdg(
                grid_type=self.grid_type(),
                meshing_args={"cell_size": h / 2},
                fracture_network=self.fracture_network,
            )

            sd_map = {}
            for sd_coarse, sd_fine in zip(
                mdg_coarse.subdomains(dim=1), mdg_fine.subdomains(dim=1)
            ):
                sd_map[sd_coarse] = sd_fine

            intf_map = {}
            for intf_coarse, intf_fine in zip(
                mdg_coarse.interfaces(dim=1), mdg_fine.interfaces(dim=1)
            ):
                intf_map[intf_coarse] = intf_fine

            mdg_coarse.replace_subdomains_and_interfaces(
                sd_map=sd_map, interface_map=intf_map
            )
            mdg_final = mdg_coarse.copy()
            pp.set_local_coordinate_projections(mdg_final)

        else:
            # The mdg is matching, and we create the mdg in the usual way
            mdg_final = pp.create_mdg(
                self.grid_type(),
                self.meshing_arguments(),
                self.fracture_network,
            )

        # Finally, we have our mdg
        self.mdg = mdg_final

        # Dimensionality of highest-dimensional manifold
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        # Create well network.
        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections.
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fractures + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures
            self.well_network.mesh(self.mdg)


class BoundaryConditions(pp.PorePyModel):

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet boundary condition to West and East sides."""
        if sd.dim == 2:
            sides = self.domain_boundary_sides(sd)
            dir_faces = sides.east + sides.west
            bc = pp.BoundaryCondition(sd, dir_faces, "dir")
        else:
            bc = pp.BoundaryCondition(sd)

        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assign unitary pressure drop between West and East sides."""
        cc_west = bg.cell_centers[0] < 1e-5
        values = np.zeros(bg.num_cells)
        values[cc_west] = 1.0
        return values


class SolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
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
        is_non_matching = self.params.get("non_matching", False)
        estimate_errors(self.mdg, is_non_matching=is_non_matching)

        # Compute error indicators
        compute_error_indicators(self.mdg)

        # Transfer from iterate to time step
        transfer_errors_iterate_solutions(self.mdg)

        # Export error indicators
        if self.params.get("export_results", False):
            self.exporter.write_vtu(
                [
                    "pressure",
                    "interface_darcy_flux",
                    "diffusive_error",
                    "residual_error",
                    "error_indicator",
                ]
            )

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def _is_time_dependent(self) -> bool:
        """The problem is stationary."""
        return False


# %% Mixer
class CrossingFracsLeftToRight2D(  # type: ignore[misc]
    Geometry,
    ErrorEstimatesSaveData,
    SolutionStrategy,
    BoundaryConditions,
    pp.SinglePhaseFlow,
):
    """Mixer class."""


@pytest.mark.parametrize("non_match", [True])
def test_model_sequence(non_match: bool):

    target_mesh_sizes = [0.2, 0.1, 0.05]
    intf_errors = []

    for h in target_mesh_sizes:

        params = {
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": h},
            "non_matching": non_match,
            "export_results": True,
            "folder_name": "l2r2d",
            "times_to_export": [],
        }
        model = CrossingFracsLeftToRight2D(params)
        pp.run_time_dependent_model(model, {})

        total = 0.0
        for intf, data in model.mdg.interfaces(dim=1, return_data=True):
            total += data["estimates"]["diffusive_error"].sum()
        intf_errors.append(np.sqrt(total))

    print("L2-like norm over 1D interfaces:", np.array(intf_errors))
