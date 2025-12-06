"""
Module containing the script to run the analysis for the third numerical example of
the paper.

Run this script to generate the `txt` files for both, uniform and adaptive mesh
refinement.

"""

from __future__ import annotations

import pickle

import mdamr
import numpy as np
import porepy as pp
from examples.flow_benchmark_3d_case_3 import solid_constants
from mdamr.estimates.error_estimation import (
    compute_error_indicators,
    get_majorant,
    transfer_errors_iterate_solutions,
)
from mdamr.examples.example_3.model import Example3Model
from porepy.utils.txt_io import TxtData, export_data_to_txt

pickle_mdg = False
print_to_console = True
refinement_levels = [0]  # [0, 1, 2, 3] are available
out_dofs = []
out_majorant = []

for refinement_level in refinement_levels:
    # Declare parameters
    params = {
        "material_constants": {"solid": solid_constants},
        "refinement_level": refinement_level,
    }

    # Run the model
    setup = Example3Model(params)
    pp.run_time_dependent_model(setup, {})
    mdg = setup.mdg
    dofs = setup.equation_system.num_dofs()

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

    for intf, d in setup.mdg.interfaces(return_data=True):
        subdomains = setup.interfaces_to_subdomains([intf])
        print("Id: ", intf.id)
        print("Dim: ", intf.dim)
        print("Num subdomain neighbors: ", len(subdomains))
        print("Subdomain dims.", "High:", subdomains[0].dim, "Low:", subdomains[1].dim)
        print(50 * "-")

    # %% Estimate errors
    mdamr.estimate_errors(
        mdg, pressure_reconstruction_method="patchwise_p1"
    )  # Assume no sources and compute residual error separately

    # Global error
    majorant = get_majorant(mdg)

    # Printing
    if print_to_console:
        print(50 * "=")
        print("Refinement level: ", refinement_level)
        print("Degrees of freedom: ", dofs)
        print("Majorant: ", majorant)
        print(50 * "=")

    # Append results to export
    out_dofs.append(dofs)
    out_majorant.append(majorant)

# %% Export to txt file
data_dofs = TxtData("dofs", np.asarray(out_dofs), "%d")
data_majorant = TxtData("majorant", np.asarray(out_majorant), "%2.2e")
list_of_data_to_export = [data_dofs, data_majorant]
export_data_to_txt(list_of_data_to_export, "error_analysis.txt")

# %% Compute error indicator and transfer errors to pp.ITERATE_SOLUTIONS
compute_error_indicators(mdg)
transfer_errors_iterate_solutions(mdg)

# %% Export results to ParaView
exporter = pp.Exporter(mdg, "out")
exporter.write_vtu(["pressure", "diffusive_error", "error_indicator"])
