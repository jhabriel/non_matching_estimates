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

import mdnme
from mdnme.examples.varela_jnum_2d.model import (
    VarelaJNum2023Setup,
    manu_incomp_fluid,
    manu_incomp_solid,
)
from mdnme.amr.pp_refinements import refine_sd_1d, refine_intf_1d


pickle_mdg = False
print_to_console = True
solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}

mesh_size = 0.05

# Export lists
out_dofs = []
out_majorant = []
out_true_error = []
out_eff_idx = []

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

# Assume only fracture grid is refined
# marked_elements = None
# new_sd_frac = refine_sd_1d(sd=sd_frac, marked_elements=None)

# Assume ony the mortar is refined
marked_elements = None
new_intf = refine_intf_1d(intf, marked_elements=marked_elements)

# Now, replace the grid in the mixed-dimensional grid
# mdg.replace_subdomains_and_interfaces(sd_map={sd_frac:new_sd_frac})
mdg.replace_subdomains_and_interfaces(intf_map={intf: new_intf})

new_sd_matrix, new_d_matrix = mdg.subdomains(return_data=True, dim=2)[0]
new_sd_frac, new_d_frac = mdg.subdomains(return_data=True, dim=1)[0]
new_intf, new_d_intf = mdg.interfaces(return_data=True, dim=1)[0]

# This approach seems to be working just fine
# The idea would then be to update all the mixed-dimensional grid in each refinement
# step. This is inneficient, but is a good first alternative.
