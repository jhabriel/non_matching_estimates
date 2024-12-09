import pickle

import porepy as pp
from model import Mono2d, manu_incomp_fluid, manu_incomp_solid

import mdamr as amr

pickle_mdg = False

# %% Setup and run the model
# Parameters
solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}
params = {
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.25 / 8},
    "manufactured_solution": "trigonometric",
    "plot_results": False,
}

# Run the model
setup = Mono2d(params)
pp.run_time_dependent_model(setup, {})
mdg = setup.mdg

# %% Export and read mdg (this is experimental for the moment)
if pickle_mdg:
    # Pickling the mdg
    print("Saving the mdg into a pickle file.")
    with open("mdg.pickle", "wb") as handle:
        pickle.dump(setup.mdg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")

    # Reloading the mdg
    with open("mdg.pickle", "rb") as handle:
        mdg: pp.MixedDimensionalGrid = pickle.load(handle)

# %% Retrieve grid and data
sd = mdg.subdomains()[0]
d = mdg.subdomain_data(sd)

# %% Estimate errors
amr.estimate_errors(mdg, pressure_reconstruction_method="patchwise_p1")  # no sources
d["estimates"]["residual_error"] = setup.exact_sol.residual_error(sd, d)

# %% Compute local errors

diffusive = (d["estimates"]["diffusive_error"].sum()) ** 0.5
residual = (d["estimates"]["residual_error"].sum()) ** 0.5
majorant = diffusive + residual

# %% Obtain true errors
# true_pressure_error = te.pressure_error()
# true_velocity_error = te.velocity_error()
# true_combined_error = true_pressure_error + true_velocity_error + residual_error
#
# # %% Compute efficiency indices
# i_eff_p = majorant_pressure / true_pressure_error
# i_eff_u = majorant_velocity / true_velocity_error
# i_eff_pu = majorant_combined / true_combined_error

print(50 * "-")
print(f"Cell size: {setup.params['meshing_arguments']['cell_size']}")
print(f"Majorant: {majorant}")
print(f"Diffusive error: {diffusive}")
print(f"Residual error: {residual}")
print(50 * "-")

# %% Error estimates
# diffusive_error = d["estimates"]["diffusive_error"]
# pp.plot_grid(sd, diffusive_error, plot_2d=True, title="Diffusive Error")
#
# residual_error = d["estimates"]["residual_error"]
# pp.plot_grid(sd, residual_error, plot_2d=True, title="Residual Error")

# %% Compute error indicators
amr.compute_error_indicators(mdg)
error_indicator_matrix = d["estimates"]["error_indicator"]
pp.plot_grid(sd, error_indicator_matrix, plot_2d=True, title="Indicator")
