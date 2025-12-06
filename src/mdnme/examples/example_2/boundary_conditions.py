"""Module containing the modified boundary conditions of the benchmark."""

import numpy as np
import porepy as pp


class BoundaryConditionsModified(pp.PorePyModel):
    """Define inlet and outlet boundary conditions as specified by the benchmark."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign Dirichlet boundary condition at outlet boundary."""
        b_faces = sd.tags["domain_boundary_faces"].nonzero()

        if b_faces != 0:
            b_faces_centers = sd.face_centers[:, b_faces]
            b_inflow = np.logical_and.reduce(
                tuple(b_faces_centers[i, :] < 0.25 - 1e-8 for i in range(3))
            )
            b_outflow = np.logical_and.reduce(
                tuple(b_faces_centers[i, :] > 0.875 + 1e-8 for i in range(3))
            )

            # Outlet faces correspond to Dirichlet boundary conditions. The rest are set
            # as Neumann by default.
            dir_faces_inflow = b_faces[0][b_inflow[0]]
            dir_faces_outflow = b_faces[0][b_outflow[0]]
            dir_faces = np.hstack((dir_faces_inflow, dir_faces_outflow))
            bc = pp.BoundaryCondition(sd, dir_faces, "dir")

        else:
            bc = pp.BoundaryCondition(sd)

        return bc

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assign unitary pressure at the outlet boundary:

        partialOmega_outlet = {x in partialOmega : x_1, x_2, x_3 > 0.875}

        """
        cc = bg.cell_centers
        nc = bg.num_cells

        faces_inflow = np.logical_and.reduce(
            tuple(cc[i, :] < 0.25 - 1e-8 for i in range(3))
        )

        faces_outflow = np.logical_and.reduce(
            tuple(cc[i, :] > 0.875 + 1e-8 for i in range(3))
        )

        val_inflow = self.units.convert_units(1, "Pa")
        val_outflow = self.units.convert_units(0, "Pa")

        values = np.zeros(nc)
        values[faces_inflow] = val_inflow
        values[faces_outflow] = val_outflow

        return values
