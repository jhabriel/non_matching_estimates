"""This module contains the convergence analysis for numerical example 3."""

import numpy as np
import porepy as pp
from porepy.utils.txt_io import TxtData, export_data_to_txt

from mdnme.estimates.error_estimation import aggregate_local_errors, get_majorant
from mdnme.examples.example_3.model import SmallFeaturesModel, solid_constants

# Loop through the matching and the non-matching case
for is_nonmatching in [False, True]:

    # Initialize lists for exporting results
    sd_error_1d = []
    sd_error_2d = []
    sd_error_3d = []
    intf_error_1d = []
    intf_error_2d = []
    majorant = []

    # Setup the model and solve using MPFA
    if not is_nonmatching:
        file_name = "matching"
    else:
        file_name = "non_matching"

    params = {
        "material_constants": {"solid": solid_constants},  # material parameters
        "refinement_level": 0,  # coarsest level
        "non_matching": is_nonmatching,  # whether to use matching or nonmatching
        "export_to_vtu": True,  # whether to export results to paraview
        "file_name": file_name,  # name of the file used to store the results
        "folder_name": "example_3",  # name of the folder used to store the results
        "times_to_export": [],  # avoid exporting in regular way
        "refinement": "nested",  # used for non-matching grid generation
        "matching_from_geo": True,  # used for matching grid generation
        "source_rate": 0.1,  # defines the magnitude of injection/production
    }
    print(f"Setting up the model for refinement level {0}.")
    model = SmallFeaturesModel(params)
    print("Done setting up the model for refinement ")

    print(f"Running model for refinement level {0}.")
    pp.run_time_dependent_model(model, params)
    print("Done running model.")

    # Compute errors of same dimensionality and the global majorant
    local_errors = aggregate_local_errors(model.mdg)
    sd_error_1d.append(local_errors["subdomain_error"][1])
    sd_error_2d.append(local_errors["subdomain_error"][2])
    sd_error_3d.append(local_errors["subdomain_error"][3])
    intf_error_1d.append(local_errors["interface_error"][1])
    intf_error_2d.append(local_errors["interface_error"][2])
    majorant.append(get_majorant(model.mdg))

    # Export results
    sd_error_1d = TxtData(header="sd_1d", array=np.asarray(sd_error_1d))
    sd_error_2d = TxtData(header="sd_2d", array=np.asarray(sd_error_2d))
    sd_error_3d = TxtData(header="sd_3d", array=np.asarray(sd_error_3d))
    intf_error_1d = TxtData(header="intf_1d", array=np.asarray(intf_error_1d))
    intf_error_2d = TxtData(header="intf_2d", array=np.asarray(intf_error_2d))
    majorant = TxtData(header="majorant", array=np.asarray(majorant))
    txt_data = [
        majorant,
        sd_error_1d,
        sd_error_2d,
        sd_error_3d,
        intf_error_1d,
        intf_error_2d,
    ]

    # Finally, export the results in a `txt` file
    print("{Exporting results to TXT file.}")
    export_data_to_txt(txt_data, f"small_features_error_{file_name}.txt")
    print("{Done exporting results to TXT file.}")
