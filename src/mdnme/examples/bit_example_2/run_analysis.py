"""This module contains the convergence analysis for numerical example 2."""

import porepy as pp
from porepy.utils.txt_io import export_data_to_txt, TxtData

import numpy as np

from mdnme.examples.bit_example_2.model import (
    Geiger3dModel,
    solid_constants_conductive,
    solid_constants_blocking
)
from mdnme.estimates.error_estimation import (
    aggregate_local_errors,
    get_majorant,
)
from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_1d,
    _interface_diffusive_error_1d_nonmatching,
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

for non_match in [True]:

    for lvl in REFINEMENT_LEVELS:

        # Setup the model and solve using MPFA
        params = {
            "material_constants": {"solid": solid_constants_conductive},
            "refinement_level": lvl,
            "non_matching": non_match,
            "times_to_export": [],
            "export_results": True,
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
    export_data_to_txt(txt_data, "geiger_3d_errors.txt")
    mdg = model.mdg.copy()


    # # ---> Debugging
    # def compare_matching_nonmatching_1d(mdg: pp.MixedDimensionalGrid):
    #     # needed by the non-matching machinery
    #     print(" id  n_cells   max|e_match|   max|e_non|   max|e_non-e_match|")
    #     print("-------------------------------------------------------------")
    #
    #     for intf, data_intf in mdg.interfaces(return_data=True):
    #         if intf.dim != 1:
    #             continue
    #
    #         sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    #         data_high = mdg.subdomain_data(sd_high)
    #         data_low  = mdg.subdomain_data(sd_low)
    #
    #         e_match = _interface_diffusive_error_1d(
    #             intf, data_intf, sd_high, data_high, sd_low, data_low
    #         )
    #         e_non = _interface_diffusive_error_1d_nonmatching(
    #             intf, data_intf, sd_high, data_high, sd_low, data_low
    #         )
    #
    #         diff_inf = np.linalg.norm(e_non - e_match, ord=np.inf)
    #         print(f"{intf.id:3d}  {intf.num_cells:7d}  "
    #               f"{np.max(e_match):12.4e}  {np.max(e_non):12.4e}  {diff_inf:14.4e}")
    #
    #
    # def inspect_interface_1d(intf: pp.MortarGrid, mdg: pp.MixedDimensionalGrid):
    #     data_intf = mdg.interface_data(intf)
    #     sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    #     data_high = mdg.subdomain_data(sd_high)
    #     data_low  = mdg.subdomain_data(sd_low)
    #
    #     e_match = _interface_diffusive_error_1d(
    #         intf, data_intf, sd_high, data_high, sd_low, data_low
    #     )
    #     e_non = _interface_diffusive_error_1d_nonmatching(
    #         intf, data_intf, sd_high, data_high, sd_low, data_low
    #     )
    #
    #     idx = np.argmax(np.abs(e_non - e_match))
    #     print(f"Interface {intf.id}, worst mortar cell {idx}:")
    #     print(f"  e_match[{idx}] = {e_match[idx]}")
    #     print(f"  e_non  [{idx}] = {e_non[idx]}")
    #     print(f"  diff           = {e_non[idx] - e_match[idx]}")

    # # usage:
    # for intf, _ in mdg.interfaces(return_data=True):
    #     if intf.dim == 1 and intf.id == 93:  # put a "hot" id here
    #         inspect_interface_1d(intf, mdg)
    #         break