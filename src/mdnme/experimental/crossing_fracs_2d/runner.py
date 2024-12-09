import pickle

import porepy as pp
from model import TwoCrossingSetup, manu_incomp_fluid, manu_incomp_solid

import mdnme as amr

pickle_mdg = False

# %% Setup and run the model
# Parameters
solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}
params = {
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.125 / 6},
}

# Run the model
setup = TwoCrossingSetup(params)
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

# %% Exploring
mdg = setup.mdg
results = setup.results[-1]

sd_2d, d_2d = mdg.subdomains(return_data=True)[0]

# %% Plot subdomain regions

pp.plot_grid(sd_2d, results.approx_p_2d, plot_2d=True, title="Approx pressure")
pp.plot_grid(sd_2d, results.exact_p_2d, plot_2d=True, title="Exact pressure")

# %%
setup.exact_sol.plot_subregions(sd_2d)

# %%
dfun = setup.exact_sol._distance_function(sd_2d)
bfun = setup.exact_sol._bubble_function(sd_2d)

# pp.plot_grid(sd_2d, dfun, plot_2d=True, title="Distance function")
# pp.plot_grid(sd_2d, bfun, plot_2d=True, title="Bubble function")

###
q_cc_x = setup.exact_sol._matrix_flux(sd_2d, which="horizontal")
q_cc_y = setup.exact_sol._matrix_flux(sd_2d, which="vertical")
q_cc = setup.exact_sol._matrix_flux(sd_2d, which="magnitude")

pp.plot_grid(sd_2d, q_cc_x, plot_2d=True, title="q_x")
pp.plot_grid(sd_2d, q_cc_y, plot_2d=True, title="q_y")
pp.plot_grid(sd_2d, q_cc, plot_2d=True, title="|q|")
