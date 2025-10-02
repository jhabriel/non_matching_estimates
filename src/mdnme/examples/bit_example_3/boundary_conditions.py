"""Module containing the modified boundary conditions mixing.

The original benchmark imposes an inlet constant Neumann flux which we have to
replace by a constant pressure boundary condition.

"""

import porepy as pp
import numpy as np


class BoundaryConditionsModified(pp.PorePyModel):
    """Define inlet and outlet boundary conditions."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet to the top and bottom  part of the north (y=y_max)
        boundary as well as the middle portion of the south (y=y_min)."""
        # Retrieve boundary sides.
        domain_sides = self.domain_boundary_sides(sd)
        # Get Dirichlet faces.
        dir_faces = np.zeros(sd.num_faces, dtype=bool)
        # top purple region from the original paper
        north_top_dir_cells = sd.face_centers[2][domain_sides.north] > (2 / 3)
        # bottom purple region from the original paper
        north_bottom_dir_faces = sd.face_centers[2][domain_sides.north] < (1 / 3)
        # blue region from the original paper
        south_middle_dif_faces = (sd.face_centers[2][domain_sides.south] < (2 / 3)) & (
                sd.face_centers[2][domain_sides.south] > (1 / 3)
        )
        # Set north faces bc-types
        dir_faces[domain_sides.north] = north_top_dir_cells + north_bottom_dir_faces
        # Set south faces bc-types
        dir_faces[domain_sides.south] = south_middle_dif_faces
        bc = pp.BoundaryCondition(sd, dir_faces, "dir")
        return bc

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Make sure all darcy fluxes are zero"""
        return np.zeros(bg.num_cells)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assign unitary pressure to the middle south (y=y_min) boundary."""
        # Retrieve boundary sides and cell centers.
        domain_sides = self.domain_boundary_sides(bg)
        cc = bg.cell_centers
        # Get inlet faces
        inlet_faces = np.zeros(bg.num_cells, dtype=bool)
        inlet_faces[domain_sides.south] = (cc[2][domain_sides.south] < (2 / 3)) & (
            cc[2][domain_sides.south] > (1 / 3)
        )
        # Assign unitary pressure
        values = np.zeros(bg.num_cells)
        values[inlet_faces] = 1  # unitary pressure
        return values
