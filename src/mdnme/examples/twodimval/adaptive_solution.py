"""
Module containing the script to run the analysis for the first numerical example of
the paper.

Run this script to generate the `txt` files for both, uniform and adaptive mesh
refinement.

"""
from __future__ import annotations

import pickle

import numpy as np
import porepy as pp
from porepy.utils.txt_io import TxtData, export_data_to_txt
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from mdnme.amr.refinement_utils import enforce_positive_orientation


import mdnme
from mdnme.examples.varela_jnum_2d.model import (
    VarelaJNumSetup2D,
    manu_incomp_fluid,
    manu_incomp_solid,
)
from mdnme.examples.varela_jnum_2d.true_errors import VarelaJNumTrueErrors2d

from mdnme.porepy_interface.fem_grid_to_porepy_grid import fem_grid_to_sd_grid_2d, fem_grid_to_sd_grid_1d
from mdnme.porepy_interface.porepy_grid_to_fem_grid import porepy_grid_to_fem_grid
from mdnme.amr.pp_refinements import refine_sd_1d, refine_intf_1d, refine_sd_2d

class VarelaJNum2023SetupAdaptive(VarelaJNumSetup2D):

    def set_geometry(self) -> None:

        # Create the geometry through domain and fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(
            self.fractures,
            self.domain
        )
        # If the AMR is off, we produce the mdg in the usual way, otherwise we pass
        # the mdg (obtained by replacing the refined grids) in the old model through
        # the params dictionary
        amr = self.params.get("amr", "off")
        if amr == "off":
            # Create a mixed-dimensional grid.
            self.mdg = pp.create_mdg(
                self.grid_type(),
                self.meshing_arguments(),
                self.fracture_network,
                **self.meshing_kwargs(),
            )
        else:
            # Mixed-dimensional grid is provided
            self.mdg = self.params["mdg"]
            # We have to make sure that the boundary grids have their geometry computed
            for sd in self.mdg.subdomains():
                bg = self.mdg.subdomain_to_boundary_grid(sd)
                bg.compute_geometry()

        # Dimensionality of highest-dimensional manifold
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)


pickle_mdg = False
print_to_console = True
solid_constants = pp.SolidConstants(**manu_incomp_solid)
fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}


params = {
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.125},
}

tol = 0.01
max_refinement_steps = 10
refinement_step = 0


#%% Declare initial mdg
model = VarelaJNum2023SetupAdaptive(params)
model.prepare_simulation()
mdg = model.mdg

sd_matrix = mdg.subdomains()[0]
coordinates = sd_matrix.nodes.transpose()[:, :sd_matrix.dim]
elements = sd_matrix.cell_node_matrix()
new_elements = enforce_positive_orientation(coordinates, elements)

# Now, we need to produce back a valid TriangleGrid
points = coordinates.T
triangles = new_elements.T
new_sd_matrix = pp.TriangleGrid(p=points, tri=triangles)
new_sd_matrix.compute_geometry()

sd_frac = mdg.subdomains()[1]
new_sd_frac = mdg.subdomains()[1]

intf = mdg.interfaces()[0]
new_intf = mdg.interfaces()[0]

#
# # Replace grid
# model.mdg.replace_subdomains_and_interfaces(
#     sd_map={sd_matrix: sd_matrix, sd_frac: sd_frac},
#     intf_map={intf: intf}
# )

# # Main AMR loop
# while refinement_step < max_refinement_steps:
#
#     # ----------------------------- STEP 1: SOLVE -------------------------------------
#     if refinement_step == 0:
#         model = VarelaJNum2023SetupAdaptive(params)
#         pp.run_time_dependent_model(model, {})
#     else:
#         params["amr"] = "on"
#         params["prepare_simulation"] = False
#         params["mdg"] = model.mdg
#         model = VarelaJNum2023SetupAdaptive(params)
#         pp.run_time_dependent_model(model, {})
#     # ---------------------------------------------------------------------------------
#
#     # ----------------------------- STEP 2: ESTIMATE ----------------------------------
#     # Retrieve subdomains and data dictionaries
#     dofs = model.mdg.num_subdomain_cells() + model.mdg.num_interface_cells()
#     sd_matrix, d_matrix = model.mdg.subdomains(return_data=True, dim=2)[0]
#     sd_frac, d_frac = model.mdg.subdomains(return_data=True, dim=1)[0]
#     intf, d_intf = model.mdg.interfaces(return_data=True, dim=1)[0]
#     # pp.save_img(f"grid_{refinement_step}.png", model.mdg, plot_2d=True)
#
#     # Print info about grids
#     print(50 * "=")
#     print(f"Old sd_primary has {sd_matrix.num_cells} dofs.")
#     print(f"Old sd_secondary has {sd_frac.num_cells} dofs.")
#     print(f"Old interface has {intf.num_cells} dofs.")
#     print()
#
#     # %% Estimate errors
#     mdamr.estimate_errors(
#         model.mdg, pressure_reconstruction_method="keilegavlen_p1"
#     )  # Assume no sources and compute residual error separately
#
#     d_matrix["estimates"]["residual_error"] = model.exact_sol.residual_error_matrix(
#         sd_matrix, d_matrix
#     )
#     d_frac["estimates"]["residual_error"] = model.exact_sol.residual_error_fracture(
#         sd_frac, d_frac
#     )
#     # Diffusive error
#     diffusive_sd_matrix = d_matrix["estimates"]["diffusive_error"]
#     diffusive_sd_frac = d_frac["estimates"]["diffusive_error"]
#     diffusive_intf_left = d_intf["estimates"]["diffusive_error"][
#         int(intf.num_cells / 2) :
#     ]
#     diffusive_intf_right = d_intf["estimates"]["diffusive_error"][
#         : int(intf.num_cells / 2)
#     ]
#
#     diffusive_error = (
#         diffusive_sd_matrix.sum()
#         + diffusive_sd_frac.sum()
#         + diffusive_intf_left.sum()
#         + diffusive_intf_right.sum()
#     ) ** 0.5
#
#     # Residual error
#     residual_sd_matrix = d_matrix["estimates"]["residual_error"]
#     residual_sd_frac = d_frac["estimates"]["residual_error"]
#     residual_error = (residual_sd_matrix.sum() + residual_sd_frac.sum()) ** 0.5
#
#     # Global error
#     majorant = diffusive_error + residual_error
#
#     # True error
#     te = Varela2023JNumTrueErrors2d()
#     true_error = te.true_error(model.mdg)
#
#     # Efficiency index
#     efficiency_index = majorant / true_error
#
#     # Printing
#     if print_to_console:
#         print(50 * "=")
#         #print("Cell size: ", mesh_size)
#         print("Degrees of freedom: ", dofs)
#         print("Majorant: ", majorant)
#         print("True error: ", true_error)
#         print("Efficiency index: ", efficiency_index)
#         print(50 * "=")
#
#     # Error indicators
#     indicator_matrix = (diffusive_sd_matrix + residual_sd_matrix) ** 0.5
#     indicator_frac = (diffusive_sd_frac + residual_sd_frac) ** 0.5
#     indicator_intf = d_intf["estimates"]["diffusive_error"] ** 0.5
#     indicators = np.concatenate((indicator_matrix, indicator_frac, indicator_intf))
#     # --------------------------------------------------------------------------------
#
#     # ------------------------------ STEP 3: MARK ------------------------------------
#     # Determined marked elements based on the error indicators.
#     # Note that the marking is done for the whole mixed-dimensional domain
#     # The consequence of this is that we may have marked elements in any subdomain
#     # and interface
#     marked_elements = mdamr.doerfler_marking(indicators, theta=0.45)
#     marked_elements_matrix = marked_elements[0:sd_matrix.num_cells]
#     marked_elements_frac = marked_elements[
#         sd_matrix.num_cells:sd_matrix.num_cells+sd_frac.num_cells
#     ]
#     marked_elements_intf = marked_elements[sd_matrix.num_cells+sd_frac.num_cells:]
#     # --------------------------------------------------------------------------------
#
#     # --------------------------- STEP 4: REFINE -------------------------------------
#     # Refine 2d subdomain
#     new_sd_matrix = refine_sd_2d(sd_matrix, marked_elements_matrix)
#
#     # Refine 1d subdomain
#     new_sd_frac = refine_sd_1d(sd_frac, marked_elements_frac)
#
#     # Refine interface
#     new_intf = refine_intf_1d(intf, marked_elements_intf)
#
#     # Replace old grids by new grids
#     model.mdg.replace_subdomains_and_interfaces(
#         sd_map={sd_matrix: new_sd_matrix},
#         #sd_map={sd_matrix: new_sd_matrix, sd_frac: new_sd_frac},
#         #intf_map={intf: new_intf}
#     )
#
#
#     for mortar in model.mdg.interfaces():
#         mortar._set_projections()
#
#     # # Re-initialize mortar mappings
#     # for mortar in model.mdg.interfaces():
#     #     # Compute primary-to-secondary mapping
#     #     primary_secondary = (
#     #             mortar.mortar_to_primary_int() @ mortar.secondary_to_mortar_int()
#     #     ).T
#     #     # Re-initialize mapping
#     #     secondary_f, primary_f, data = sparse_array_to_row_col_data(primary_secondary)
#     #     if mortar.num_sides() == 2:
#     #         # After the above sorting, we know that the faces on the other side is in
#     #         # the second half of primary_f, also if face_duplicate_ind is given.
#     #         # ASSUMPTION: The mortar grids on the two sides should match each other, or
#     #         # else, the below indexing is wrong. This also means that the size of
#     #         # primary_f is an even number.
#     #         sz = int(primary_f.size / 2)
#     #         mortar._ind_face_on_other_side = primary_f[sz:]
#
#     # Print refined info
#     # print(f"New sd_primary has {new_sd_matrix.num_cells} dofs.")
#     # print(f"New sd_secondary has {new_sd_frac.num_cells} dofs.")
#     # print(f"New interface has {new_intf.num_cells} dofs.")
#     # print(50 * "=")
#     # print()
#
#     # --------------------------------------------------------------------------------
#
#     # INCREMENT COUNTER
#     refinement_step += 1
