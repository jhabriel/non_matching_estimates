import pickle

import numpy as np
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
    "meshing_arguments": {"cell_size": 0.125 / 2},
    "plot_results": True,
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

# Subdomains
sd_2d, d_2d = mdg.subdomains(return_data=True)[0]
sd_1d_west, d_1d_west = mdg.subdomains(return_data=True)[1]
sd_1d_east, d_1d_east = mdg.subdomains(return_data=True)[2]
sd_1d_south, d_1d_south = mdg.subdomains(return_data=True)[3]
sd_1d_north, d_1d_north = mdg.subdomains(return_data=True)[4]
sd_0d, d_0d = mdg.subdomains(return_data=True)[5]

# Exact pressure
p_2d_ex = setup.results[-1].exact_p_2d
p_1d_west_ex = setup.results[-1].exact_p_1d_west
p_1d_east_ex = setup.results[-1].exact_p_1d_east
p_1d_south_ex = setup.results[-1].exact_p_1d_south
p_1d_north_ex = setup.results[-1].exact_p_1d_north
p_0d_ex = setup.results[-1].exact_p_0d

# Approximate pressure
p_2d_fv = setup.results[-1].approx_p_2d
p_1d_west_fv = setup.results[-1].approx_p_1d_west
p_1d_east_fv = setup.results[-1].approx_p_1d_east
p_1d_south_fv = setup.results[-1].approx_p_1d_south
p_1d_north_fv = setup.results[-1].approx_p_1d_north
p_0d_fv = setup.results[-1].approx_p_0d
