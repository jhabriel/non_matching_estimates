"""Module for running the third numerical example"""
import mdnme
import porepy as pp
import numpy as np

from mdnme.examples.bit_example_3.model import SmallFeaturesModel, solid_constants
from mdnme.estimates.error_estimation import (
    compute_sd_and_intf_errors_of_equal_dim,
    get_majorant,
)
from porepy.utils.txt_io import export_data_to_txt, TxtData

# Initialize lists for exporting results
sd_error_1d = []
sd_error_2d = []
sd_error_3d = []
intf_error_1d = []
intf_error_2d = []
majorant = []

# REFINEMENT_LEVELS = [0, 1, 2, 3]
REFINEMENT_LEVELS = [1]

for lvl in REFINEMENT_LEVELS:

    # Setup the model and solve using MPFA
    params = {
        "material_constants": {"solid": solid_constants},
        "refinement_level": lvl,
        "non_matching": True,
        "export_to_vtu": False,
    }
    print(f"Setting up the model for refinement level {lvl}.")
    model = SmallFeaturesModel(params)
    print(f"Done setting up the model for refinement ")

    print(f"Running model for refinement level {lvl}.")
    pp.run_time_dependent_model(model, params)
    print(f"Done running model.")

    # Estimate the errors
    print(f"Estimating errors for refinement level {lvl}.")
    mdnme.estimate_errors(mdg=model.mdg, non_matching_nested=False)
    print(f"Done estimating errors.")

    # Store errors in the corresponding lists

    # Compute errors of the same dimensionality
    errors_of_same_dim = compute_sd_and_intf_errors_of_equal_dim(model.mdg)
    sd_error_1d.append(errors_of_same_dim['subdomain_error'][1])
    sd_error_2d.append(errors_of_same_dim['subdomain_error'][2])
    sd_error_3d.append(errors_of_same_dim['subdomain_error'][3])
    intf_error_1d.append(errors_of_same_dim['interface_error'][1])
    intf_error_2d.append(errors_of_same_dim['interface_error'][2])
    majorant.append(get_majorant(model.mdg))

# Export results
sd_error_1d = TxtData(header="sd_1d", array=np.asarray(sd_error_1d))
sd_error_2d = TxtData(header="sd_2d", array=np.asarray(sd_error_2d))
sd_error_3d = TxtData(header="sd_3d", array=np.asarray(sd_error_3d))
intf_error_1d = TxtData(header="intf_1d", array=np.asarray(intf_error_1d))
intf_error_2d = TxtData(header="intf_2d", array=np.asarray(intf_error_2d))
majorant = TxtData(header="majorant", array=np.asarray(majorant))
txt_data = [
    sd_error_1d,
    sd_error_2d,
    sd_error_3d,
    intf_error_1d,
    intf_error_2d,
]
export_data_to_txt(txt_data, "small_features_error.txt")
