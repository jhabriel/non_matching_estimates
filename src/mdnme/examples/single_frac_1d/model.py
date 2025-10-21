"""2D/1D flow with a tilted fracture. Flow left to right."""


import porepy as pp
import numpy as np
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.fracs.fracture_network_3d import FractureNetwork3d

from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow

import mdnme
from mdnme.estimates.error_estimation import compute_error_indicators, transfer_errors_iterate_solutions
from mdnme.utils.grid_rotation import assign_canonical_rotations


class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.units.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.units.convert_units(
            np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a diagonal fracture we cannot use a cartesian grid.
        Cartesian grid is the default grid type, and we therefore override this method to assign simplex instead.

        """
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        cell_size = self.units.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def set_geometry(self) -> None:
        self.set_domain()
        self.set_fractures()
        self.create_fracture_network()
        self.create_mdg()
        self.nd = self.mdg.dim_max()
        pp.set_local_coordinate_projections(self.mdg)
        assign_canonical_rotations(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)


class NonMatching2D1DGeometry:
    """Build a 2D/1D tilted-fracture mdg, optionally non-matching by refining only the 1D fracture."""

    def set_domain(self) -> None:
        size = self.units.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        # Diagonal fracture from (0.2, 0.2) to (1.8, 1.8) in meters.
        # NOTE: This is in *physical* units; PorePy will handle scaling.
        pts = self.units.convert_units(np.array([[0.2, 1.8], [0.2, 1.8]]), "m")
        self._fractures = [pp.LineFracture(pts)]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def _build_matching_mdg(self, h: float) -> pp.MixedDimensionalGrid:
        mesh_args = {"cell_size": self.units.convert_units(h, "m")}
        return pp.create_mdg(self.grid_type(), mesh_args, self.fracture_network)

    def _make_nonmatching_by_replacing_1d(
        self, mdg_matrix: pp.MixedDimensionalGrid, mdg_with_fine_frac: pp.MixedDimensionalGrid
    ) -> pp.MixedDimensionalGrid:
        """Replace 1D fracture subdomains in mdg_matrix with the fine fracture grids from mdg_with_fine_frac."""
        mdg_nm = mdg_matrix.copy()

        # Map “same” 1D subdomain positions from the two MDGs; they are yielded in consistent order.
        sd_map = {}
        for sd_coarse, sd_fine in zip(mdg_nm.subdomains(), mdg_with_fine_frac.subdomains()):
            if sd_coarse.dim != sd_fine.dim:
                # Defensive: should not happen if both MDGs built from same FN.
                continue
            if sd_coarse.dim == 1:
                sd_map[sd_coarse] = sd_fine

        # Replace fracture subdomains + rewire interfaces automatically
        mdg_nm.replace_subdomains_and_interfaces(sd_map=sd_map)

        # Rebuild local coord projections and rotations for *all* subdomains/interfaces
        pp.set_local_coordinate_projections(mdg_nm)
        assign_canonical_rotations(mdg_nm)
        return mdg_nm

    def set_geometry(self) -> None:
        """Construct matching or non-matching mdg depending on parameters.

        Params recognized:
          - "h_matrix": coarse matrix target size (default 0.25 m)
          - "h_fracture": fracture target size for fine pass (default 0.10 m)
          - "non_matching": bool (default False). If True, refine only 1D fracture.
        """
        self.set_domain()
        self.set_fractures()
        self.create_fracture_network()

        # Defaults
        h_matrix = float(self.params.get("h_matrix", 0.25))
        h_fract  = float(self.params.get("h_fracture", 0.10))
        non_matching = bool(self.params.get("non_matching", False))

        # 1) coarse, matching
        mdg_coarse = self._build_matching_mdg(h_matrix)

        if not non_matching:
            self.mdg = mdg_coarse
        else:
            # 2) fine (we’ll steal only the 1D subdomain)
            mdg_fine_all = self._build_matching_mdg(h_fract)
            self.mdg = self._make_nonmatching_by_replacing_1d(mdg_coarse, mdg_fine_all)

        self.nd = self.mdg.dim_max()

        # Store domain / fracture network for consistency with PorePy models
        self._domain = self.fracture_network.domain
        self._fractures = self.fracture_network.fractures

        # Set wells if present
        self.set_well_network()
        if len(self.well_network.wells) > 0:
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(self.well_network, self.fracture_network)
            self.well_network.mesh(self.mdg)


class ModifiedBC(BoundaryConditionsSinglePhaseFlow):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.west + domain_sides.east, "dir")
        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        # See section on scaling for explanation of the conversion.
        values[domain_sides.west] = self.units.convert_units(5, "Pa")
        values[domain_sides.east] = self.units.convert_units(2, "Pa")
        return values


class SinglePhaseFlowModifiedSolutionStrategy(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Modified solution strategy for the verification setup."""

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        # Save error estimates data
        self.error_estimates_data_saving()


class SinglePhaseFlow2dSingleFrac(
    NonMatching2D1DGeometry,
    ModifiedBC,
    SinglePhaseFlowModifiedSolutionStrategy,
    mdnme.ErrorEstimatesSaveData,
    SinglePhaseFlow):
    """Adding both geometry and modified boundary conditions to the default model."""
    ...


#%% Runner

fluid_constants = pp.FluidComponent(viscosity=0.1, density=0.2)
solid_constants = pp.SolidConstants(permeability=0.5, porosity=0.25)
material_constants = {"fluid": fluid_constants, "solid": solid_constants}
model_params = {
    "material_constants": material_constants,
    "grid_type": "simplex",
    "h_matrix": 0.25,  # coarse matrix side
    "h_fracture": 0.08,  # finer fracture size
    "non_matching": True,  # flip to False to get a fully matching mdg
}

model = SinglePhaseFlow2dSingleFrac(model_params)
pp.run_time_dependent_model(model)
mdnme.estimate_errors(model.mdg)
compute_error_indicators(model.mdg)
transfer_errors_iterate_solutions(model.mdg)


pp.save_img(
    "griddy.png",
    model.mdg,
    "diffusive_error",
    figsize=(10, 8),
    linewidth=0.25,
    title="Pressure distribution",
    plot_2d=True
)