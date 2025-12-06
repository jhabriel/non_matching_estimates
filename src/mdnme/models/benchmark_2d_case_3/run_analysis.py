"""
Module containing the script to run the analysis for the second numerical example of
the paper.

Run this script to generate the `txt` files for both, uniform and adaptive mesh
refinement.

"""
from __future__ import annotations

import porepy as pp

# from mdamr.examples.benchmark_2d_case_3.model import (
#     Flemisch2018Case3Model,
#     solid_constants,
# )
from mdnme.models.benchmark_2d_case_3 import FlowBenchmark2dCase3bModel, solid_constants

pickle_mdg = False
print_to_console = False
mesh_sizes = [0.1]

# Export lists
out_dofs = []
out_majorant = []

# for mesh_size in mesh_sizes:
mesh_size = 0.1

# Parameters
params = {
    "grid_type": "simplex",
    "material_constants": {"solid": solid_constants},
    "meshing_arguments": {"cell_size": mesh_size},
}

# Run the model
setup = FlowBenchmark2dCase3bModel(params)
pp.run_time_dependent_model(setup, {})
mdg = setup.mdg
dofs = setup.equation_system.num_dofs()

# # %% Export and read mdg (this is experimental for the moment)
# if pickle_mdg:
#     # Pickling the mdg
#     print("Saving the mdg into a pickle file.")
#     with open("mdg.pickle", "wb") as handle:
#         pickle.dump(setup.mdg, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("Done.")
#
#     # Reloading the mdg
#     with open("mdg.pickle", "rb") as handle:
#         mdg = pickle.load(handle)

# # %% Estimate errors
# mdamr.estimate_errors(
#     mdg, pressure_reconstruction_method="keilegavlen_p1"
# )  # Assume no sources and compute residual error separately
#
# # Global error
# majorant = get_majorant(mdg)
#
# # Printing
# if print_to_console:
#     print(50 * "=")
#     print("Cell size: ", mesh_size)
#     print("Degrees of freedom: ", dofs)
#     print("Majorant: ", majorant)
#     print(50 * "=")
#
# # Append results to export
# out_dofs.append(dofs)
# out_majorant.append(majorant)

# %% Checking
for intf, d in setup.mdg.interfaces(return_data=True):
    subdomains = setup.interfaces_to_subdomains([intf])
    print("Id: ", intf.id)
    print("Dim: ", intf.dim)
    print("Subdomain dims", subdomains[0].dim, subdomains[1].dim)
    print("Num subdomain neighbors: ", len(subdomains))
    print(50 * "-")

# %% Export to txt filedata_dofs = TxtData("dofs", np.asarray(out_dofs), "%d")
# # data_majorant = TxtData("majorant", np.asarray(out_majorant), "%2.2e")
# # list_of_data_to_export = [data_dofs, data_majorant]
# # export_data_to_txt(list_of_data_to_export, "error_analysis.txt")
# #
# # # %% Compute error indicator and transfer errors to pp.ITERATE_SOLUTIONS
# # compute_error_indicators(mdg)
# # transfer_errors_iterate_solutions(mdg)
# #
# # # %% Export results to ParaView
# # exporter = pp.Exporter(mdg, "out")
# # exporter.write_vtu(["pressure", "diffusive_error", "error_indicator"])
# #
# #
# # # %%
# # p_num = setup.pressure(setup.mdg.subdomains()).value(setup.equation_system)
# # print("Norm of mixed-dimensional pressure:", np.linalg.norm(p_num))
#
