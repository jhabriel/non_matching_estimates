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

import mdnme
from mdnme.examples.varela_jnum_2d.model import (
    VarelaJNum2023Setup,
    manu_incomp_fluid,
    manu_incomp_solid,
)
from mdnme.examples.varela_jnum_2d.true_errors import Varela2023JNumTrueErrors2d

pickle_mdg = False
print_to_console = True
solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}

mesh_sizes = [0.125, 0.125 / 2, 0.125 / 4, 0.125 / 16, 0.125 / 32]
mesh_sizes = [0.05]

# Export lists
out_dofs = []
out_majorant = []
out_true_error = []
out_eff_idx = []

for mesh_size in mesh_sizes:
    # Parameters
    params = {
        "grid_type": "simplex",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": mesh_size},
    }

    # Run the model
    setup = VarelaJNum2023Setup(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg
    dofs = setup.equation_system.num_dofs()

    # Retrieve subdomains and data dictionaries
    sd_matrix, d_matrix = mdg.subdomains(return_data=True, dim=2)[0]
    sd_frac, d_frac = mdg.subdomains(return_data=True, dim=1)[0]
    intf, d_intf = mdg.interfaces(return_data=True, dim=1)[0]

    # %% Export and read mdg (this is experimental for the moment)
    if pickle_mdg:
        # Pickling the mdg
        print("Saving the mdg into a pickle file.")
        with open("mdg.pickle", "wb") as handle:
            pickle.dump(setup.mdg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done.")

        # Reloading the mdg
        with open("mdg.pickle", "rb") as handle:
            mdg = pickle.load(handle)

    # %% Estimate errors
    mdnme.estimate_errors(
        mdg, pressure_reconstruction_method="keilegavlen_p1"
    )  # Assume no sources and compute residual error separately

    d_matrix["estimates"]["residual_error"] = setup.exact_sol.residual_error_matrix(
        sd_matrix, d_matrix
    )
    d_frac["estimates"]["residual_error"] = setup.exact_sol.residual_error_fracture(
        sd_frac, d_frac
    )
    # Diffusive error
    diffusive_sd_matrix = d_matrix["estimates"]["diffusive_error"]
    diffusive_sd_frac = d_frac["estimates"]["diffusive_error"]
    diffusive_intf_left = d_intf["estimates"]["diffusive_error"][
        int(intf.num_cells / 2) :
    ]
    diffusive_intf_right = d_intf["estimates"]["diffusive_error"][
        : int(intf.num_cells / 2)
    ]

    diffusive_error = (
        diffusive_sd_matrix.sum()
        + diffusive_sd_frac.sum()
        + diffusive_intf_left.sum()
        + diffusive_intf_right.sum()
    ) ** 0.5

    # Residual error
    residual_sd_matrix = d_matrix["estimates"]["residual_error"]
    residual_sd_frac = d_frac["estimates"]["residual_error"]
    residual_error = (residual_sd_matrix.sum() + residual_sd_frac.sum()) ** 0.5

    # Global error
    majorant = diffusive_error + residual_error

    # True error
    te = Varela2023JNumTrueErrors2d()
    true_error = te.true_error(mdg)

    # Efficiency index
    efficiency_index = majorant / true_error

    # Printing
    if print_to_console:
        print(50 * "=")
        print("Cell size: ", mesh_size)
        print("Degrees of freedom: ", dofs)
        print("Majorant: ", majorant)
        print("True error: ", true_error)
        print("Efficiency index: ", efficiency_index)
        print(50 * "=")

    # Append results to export
    out_dofs.append(dofs)
    out_majorant.append(majorant)
    out_true_error.append(true_error)
    out_eff_idx.append(efficiency_index)

# %% Export to txt file
data_dofs = TxtData("dofs", np.asarray(out_dofs), "%d")
data_majorant = TxtData("majorant", np.asarray(out_majorant), "%2.2e")
data_true_error = TxtData("true_error", np.asarray(out_true_error), "%2.2e")
data_eff_idx = TxtData("eff_index", np.asarray(out_eff_idx), "%2.4f")
list_of_data_to_export = [data_dofs, data_majorant, data_true_error, data_eff_idx]
export_data_to_txt(list_of_data_to_export, "error_analysis.txt")
