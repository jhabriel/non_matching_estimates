"""
Runner for first numerical example
"""

import porepy as pp
import numpy as np
import mdnme


from mdnme.examples.varela_jnum_3d.model import VarelaJNumSetup3D
from mdnme.examples.varela_jnum_2d.model import (
    manu_incomp_fluid,
    manu_incomp_solid,
)
from mdnme.examples.varela_jnum_3d.true_errors import VarelaJNumTrueErrors3D
from mdnme.estimates.error_estimation import estimate_errors
from mdnme.utils.transfer_grid import TransferGrid

# ---> Setup matching case as reference
solid_constants = pp.SolidConstants(**manu_incomp_solid)
fluid_constants = pp.FluidComponent(**manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}

matching_params = {
    'grid_type': 'simplex',
    'material_constants': material_constants,
    'meshing_arguments': {"cell_size": 0.125},
    "times_to_export": [],  # Supress outputs for tests
}
setup_matching = VarelaJNumSetup3D(matching_params)
pp.run_time_dependent_model(setup_matching, {})
estimate_errors(setup_matching.mdg)

non_matching_params = {
    'grid_type': 'simplex',
    'material_constants': material_constants,
    'meshing_arguments': {"cell_size": 0.125},
    'times_to_export': [],  # Supress outputs for tests
    'non_matching': True,  # used by the `set_geometry()`
    'perturb_fracture': True,  # perturb only the fracture grid
    'perturb_mortar': False,
    'refine_fracture': False,
    'refine_mortar': False,
    'translation_vector': (0, 1, 1),  # Use to translate the internal nodes
}
setup_nonmatching = VarelaJNumSetup3D(non_matching_params)
pp.run_time_dependent_model(setup_nonmatching, {})
estimate_errors(setup_nonmatching.mdg)

# Compare things...
mdg_match = setup_matching.mdg
mdg_nonmatch = setup_nonmatching.mdg

sd_mat_match, d_mat_match = mdg_match.subdomains(dim=3, return_data=True)[0]
sd_mat_nonmatch, d_mat_nonmatch = mdg_nonmatch.subdomains(dim=3, return_data=True)[0]
sd_frac_match, d_frac_match = mdg_match.subdomains(dim=2, return_data=True)[0]
sd_frac_nonmatch, d_frac_nonmatch = mdg_nonmatch.subdomains(dim=2, return_data=True)[0]
intf_match, d_intf_match = mdg_match.interfaces(dim=2, return_data=True)[0]
intf_nonmatch, d_intf_nonmatch = mdg_nonmatch.interfaces(dim=2, return_data=True)[0]

# Subdomain and interface diffusive and residual errors
diff_sd_mat_match = d_mat_match["estimates"]["diffusive_error"]
diff_sd_mat_nonmatch = d_mat_nonmatch["estimates"]["diffusive_error"]

diff_sd_frac_match = d_frac_match["estimates"]["diffusive_error"]
diff_sd_frac_nonmatch = d_frac_nonmatch["estimates"]["diffusive_error"]

diff_intf_match = d_intf_match["estimates"]["diffusive_error"]
diff_intf_nonmatch = d_intf_nonmatch["estimates"]["diffusive_error"]
diff_intf_match_left = diff_intf_match[int(intf_match.num_cells / 2) :]
diff_intf_match_right = diff_intf_match[: int(intf_match.num_cells / 2)]
diff_intf_nonmatch_left = diff_intf_nonmatch[int(intf_nonmatch.num_cells / 2) :]
diff_intf_nonmatch_right = diff_intf_nonmatch[: int(intf_nonmatch.num_cells / 2)]

resi_sd_mat_match = setup_matching.exact_sol.residual_error_matrix(
    sd_mat_match, d_mat_match
).sum()
resi_sd_mat_nonmatch = setup_nonmatching.exact_sol.residual_error_matrix(
    sd_mat_nonmatch, d_mat_nonmatch
).sum()
resi_sd_frac_match = setup_matching.exact_sol.residual_error_fracture(
    sd_frac_match, d_frac_match
).sum()
resi_sd_frac_nonmatch = setup_nonmatching.exact_sol.residual_error_fracture(
    sd_frac_nonmatch, d_frac_nonmatch
).sum()

# Global errors
diff_error_match = (diff_sd_mat_match.sum()
                    + diff_sd_frac_match.sum()
                    + diff_intf_match_left.sum()
                    + diff_intf_match_right.sum()
                    ) ** 0.5
diff_error_nonmatch = (diff_sd_mat_nonmatch.sum()
                       + diff_sd_frac_nonmatch.sum()
                       + diff_intf_nonmatch_left.sum()
                       + diff_intf_nonmatch_right.sum()
                       ) ** 0.5

residual_error_match = (resi_sd_mat_match.sum()
                        + resi_sd_frac_match.sum()
                        ) ** 0.5
residual_error_nonmatch = (resi_sd_mat_nonmatch.sum()
                           + resi_sd_frac_nonmatch.sum()
                           ) ** 0.5

majorant_match = diff_error_match + residual_error_match
majorant_nonmatch = diff_error_nonmatch + residual_error_nonmatch

# True errors
te_match = VarelaJNumTrueErrors3D(setup_matching)
te_nonmatch = VarelaJNumTrueErrors3D(setup_nonmatching)

true_error_match = te_match.true_error()
true_error_nonmatch = te_nonmatch.true_error()

eff_idx_match = majorant_match / true_error_match
eff_idx_nonmatch = majorant_nonmatch / true_error_nonmatch


