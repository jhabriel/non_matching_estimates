"""This module contains the convergence analysis for numerical example 2."""

import porepy as pp
from porepy.utils.txt_io import export_data_to_txt, TxtData

import numpy as np

from mdnme.examples.example_2.model import (
    Geiger3dModel,
    solid_constants_conductive,
)
from mdnme.estimates.error_estimation import (
    aggregate_local_errors,
    get_majorant,
)

# Initialize lists for exporting results
sd_error_1d = []
sd_error_2d = []
sd_error_3d = []
intf_error_0d = []
intf_error_1d = []
intf_error_2d = []
majorant = []

# Define refinement levels
REFINEMENT_LEVELS = [0, 1, 2]

for non_match in [True]:  # You can also add False to check the matching estimators

    for lvl in REFINEMENT_LEVELS:

        # Setup the model and solve using MPFA
        params = {
            "material_constants": {"solid": solid_constants_conductive},
            "refinement_level": lvl,
            "non_matching": non_match,
            "times_to_export": [],
            "export_results": True,
            "folder_name": "geiger3d",
        }
        print(f"Setting up the model for refinement level {lvl}.")
        model = Geiger3dModel(params)
        print(f"Done setting up the model for refinement ")

        print(f"Running model for refinement level {lvl}.")
        pp.run_time_dependent_model(model, params)
        print(f"Done running model.")

        # Compute errors of same dimensionality and the global majorant
        local_errors = aggregate_local_errors(model.mdg)
        sd_error_1d.append(local_errors['subdomain_error'][1])
        sd_error_2d.append(local_errors['subdomain_error'][2])
        sd_error_3d.append(local_errors['subdomain_error'][3])
        intf_error_0d.append(local_errors['interface_error'][0])
        intf_error_1d.append(local_errors['interface_error'][1])
        intf_error_2d.append(local_errors['interface_error'][2])
        majorant.append(get_majorant(model.mdg))

    # Export results
    sd_error_1d = TxtData(header="sd_1d", array=np.asarray(sd_error_1d))
    sd_error_2d = TxtData(header="sd_2d", array=np.asarray(sd_error_2d))
    sd_error_3d = TxtData(header="sd_3d", array=np.asarray(sd_error_3d))
    intf_error_0d = TxtData(header="intf_0d", array=np.asarray(intf_error_0d))
    intf_error_1d = TxtData(header="intf_1d", array=np.asarray(intf_error_1d))
    intf_error_2d = TxtData(header="intf_2d", array=np.asarray(intf_error_2d))
    majorant = TxtData(header="majorant", array=np.asarray(majorant))
    txt_data = [
        majorant,
        sd_error_1d,
        sd_error_2d,
        sd_error_3d,
        intf_error_0d,
        intf_error_1d,
        intf_error_2d,
    ]

    # Finally, export the results in a `txt` file
    export_data_to_txt(txt_data, "geiger_3d_errors.txt")
