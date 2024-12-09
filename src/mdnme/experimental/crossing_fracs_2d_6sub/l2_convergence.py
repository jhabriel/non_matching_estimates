import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from model import TwoCrossingSetup, manu_incomp_fluid, manu_incomp_solid
from porepy.applications.convergence_analysis import ConvergenceAnalysis

# %%

# Parameters
solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}
params = {
    "grid_type": "simplex",
    "material_constants": material_constants,
    "meshing_arguments": {"cell_size": 0.125},
    "plot_results": False,
}


# %%
cao = ConvergenceAnalysis(
    model_class=TwoCrossingSetup,
    model_params=params,
    levels=6,
    spatial_refinement_rate=2,
    temporal_refinement_rate=1,
)

lor = cao.run_analysis()

# %% Errors
cao.export_errors_to_txt(lor)
ooc = cao.order_of_convergence(lor)
