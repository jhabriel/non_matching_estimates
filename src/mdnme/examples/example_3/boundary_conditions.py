"""Module containing the modified boundary conditions mixing.

The original benchmark imposes an inlet constant Neumann flux which we have to
replace by a constant pressure boundary condition.

"""

import numpy as np
import porepy as pp


class NoFluxBoundaryConditions(pp.PorePyModel):
    """Define inlet and outlet boundary conditions."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Force no-flux boundary conditions at all sides."""
        cc = sd.face_centers
        south = cc[1] < 1e-5
        north = cc[1] > np.max(cc[1]) - 1e-5
        dir_faces = south + north
        if sd.dim == 3:
            return pp.BoundaryCondition(sd, faces=dir_faces, cond="dir")
        else:
            return pp.BoundaryCondition(sd)

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Make sure all darcy fluxes are zero."""
        return np.zeros(bg.num_cells)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Make sure all darcy fluxes are zero."""
        cc = bg.cell_centers
        south = cc[1] < 1e-5
        vals = np.zeros(bg.num_cells)
        vals[south] = 0
        return vals


class ModifiedBalanceEquation(pp.fluid_mass_balance.FluidMassBalanceEquations):
    """Modify balance equation to account for external sources."""

    def _matches_surface(
        self, sd: pp.Grid, vertices: np.ndarray, tol: float = 1e-6
    ) -> bool:
        """Check if the 2D grid sd has all given vertices among its nodes (up to tol)."""
        assert sd.dim == 2

        nodes = sd.nodes  # shape (3, n_nodes)
        verts = np.asarray(vertices, dtype=float).reshape(-1, 3)  # (n_verts, 3)

        for v in verts:
            # squared distance to all nodes
            diff = nodes.T - v  # (n_nodes, 3)
            d2 = np.sum(diff**2, axis=1)
            if np.min(d2) > tol**2:
                return False
        return True

    def _injector_idx(self, sd: pp.Grid):
        assert sd.dim == 2

        injector_vertices = np.array(
            [
                [0.05, 1.0, 0.5],
                [0.95, 1.0, 0.5],
                [0.95, 2.2, 0.85],
                [0.05, 2.2, 0.85],
            ]
        )

        if self._matches_surface(sd, injector_vertices, tol=1e-6):
            cc_centroid = injector_vertices.mean(axis=0)  # [0.5, 1.60, 0.675]
            cell_idx, distances = sd.closest_cell(cc_centroid.reshape(3, 1), True)
            cell_idx_unique = cell_idx[np.argmin(cell_idx)]
            sd_id = sd.id
        else:
            cell_idx_unique = np.nan
            sd_id = -1

        return sd_id, cell_idx_unique

    def _productor_idx(self, sd: pp.Grid):
        assert sd.dim == 2

        productor_vertices = np.array(
            [
                [0.05, 0.25, 0.5],
                [0.95, 0.25, 0.5],
                [0.95, 2.0, 0.5],
                [0.05, 2.0, 0.5],
            ]
        )

        if self._matches_surface(sd, productor_vertices, tol=1e-6):
            target_location = np.array([0.50, 0.625, 0.50])
            cell_idx, distances = sd.closest_cell(target_location.reshape(3, 1), True)
            cell_idx_unique = cell_idx[np.argmin(cell_idx)]
            sd_id = sd.id
        else:
            cell_idx_unique = np.nan
            sd_id = -1

        return sd_id, cell_idx_unique

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
            if sd.dim == 2:

                sd_id_inj, cell_idx_inj = self._injector_idx(sd)
                sd_id_prd, cell_idx_prd = self._productor_idx(sd)

                val_loc = np.zeros(sd.num_cells)

                rate = self.params.get("source_rate", 1)

                if sd.id == sd_id_inj:
                    val_loc[cell_idx_inj] = -rate
                    print(f"Injector cell coo {sd.cell_centers[:, cell_idx_inj]}")

                if sd.id == sd_id_prd:
                    val_loc[cell_idx_prd] = rate
                    print(f"Productor cell coo {sd.cell_centers[:, cell_idx_prd]}")

                values.append(val_loc)
            else:
                values.append(np.zeros(sd.num_cells))

        external_sources = pp.wrap_as_dense_ad_array(np.hstack(values))

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source
