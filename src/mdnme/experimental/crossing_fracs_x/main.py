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
    "meshing_arguments": {"cell_size": 0.125 / 8},
    "plot_results": True,
}

# Run the model
setup = TwoCrossingSetup(params)
pp.run_stationary_model(setup, {})
mdg = setup.mdg
