"""Debugging error with one-dimensional diffusive errors"""

import porepy as pp
import numpy as np
import mdnme

from mdnme.examples.bit_example_2.model import (
    Geiger3dModel,
    solid_constants_conductive,
    solid_constants_blocking
)
from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_1d_nonmatching,
    _interface_diffusive_error_2d_nonmatching,
)

    # Setup the model and solve using MPFA
params = {
    "material_constants": {"solid": solid_constants_conductive},
    "refinement_level": 0,
    "non_matching": True,
    "times_to_export": [],
    "export_results": False,
}
print(f"Setting up the model for refinement level {0}.")
model = Geiger3dModel(params)
print(f"Done setting up the model for refinement ")

print(f"Running model for refinement level {0}.")
pp.run_time_dependent_model(model, params)
print(f"Done running model.")


# ----> DEBUGGING <-----

# Insert constant data

mdg = model.mdg
C = 1.0

for sd, d in mdg.subdomains(return_data=True):

    if sd.dim == 2:
        d["estimates"]["recon_sd_pressure"] = np.tile(
            [0.0, 0.0, C], (sd.num_cells, 1)
        )
    elif sd.dim == 1:
        d["estimates"]["recon_sd_pressure"] = np.tile(
            [0.0, C], (sd.num_cells, 1)
        )
    else:
        continue

for intf, d in mdg.interfaces(return_data=True):
        d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)
        d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0


# ---> Measure errors <----

current_max = 0

for intf, data_intf in mdg.interfaces(return_data=True):

    if intf.dim == 1:

        sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
        data_high = mdg.subdomain_data(sd_high)
        data_low = mdg.subdomain_data(sd_low)

        diff = _interface_diffusive_error_1d_nonmatching(
                intf, data_intf, sd_high, data_high, sd_low, data_low
        )
        print("max interface diffusive error:", diff.max())
        if current_max < diff.max():
            current_max = diff.max()
            id_max = intf.id

# ---> DEBUGGING: GLOBAL LINEAR PRESSURE <-------

# Choose a global affine field p(x) = r · x + c
r = np.array([1.0, 0.5, -0.3])   # arbitrary nontrivial direction in R^3
c0 = 0.7                         # arbitrary intercept

for sd, d in mdg.subdomains(return_data=True):

    # Skip 0D subdomains
    if sd.dim == 0:
        continue

    # Rotated grid for this subdomain
    g_rot = mdnme.RotatedGrid(sd)
    R = g_rot.rotation_matrix       # 3x3
    dim_bool = np.array(g_rot.dim_bool, dtype=bool)  # length 3

    # Global gradient in this grid's local coordinates
    r_loc = R @ r                   # still length 3
    r_act = r_loc[dim_bool]         # length = sd.dim

    if sd.dim == 2:
        # P1 coeffs [a_x, a_y, c] in local 2D coords
        a_x, a_y = r_act
        d["estimates"]["recon_sd_pressure"] = np.tile(
            [a_x, a_y, c0], (sd.num_cells, 1)
        )
    elif sd.dim == 1:
        # P1 coeffs [a_s, c] in local 1D coord
        (a_s,) = r_act
        d["estimates"]["recon_sd_pressure"] = np.tile(
            [a_s, c0], (sd.num_cells, 1)
        )
    else:
        # You can skip 3D here if this model only uses 2D-1D coupling,
        # or extend similarly with 3 local components.
        continue

# Zero interface flux, unit permeability
for intf, d in mdg.interfaces(return_data=True):
    d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)
    d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0


# ---> Measure errors <----

current_max = 0

for intf, data_intf in mdg.interfaces(return_data=True):

    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low = mdg.subdomain_data(sd_low)

    if intf.dim == 2:
        continue
        diff = _interface_diffusive_error_2d_nonmatching(
            intf, data_intf, sd_high, data_high, sd_low, data_low
        )
    elif intf.dim == 1:

        diff = _interface_diffusive_error_1d_nonmatching(
                intf, data_intf, sd_high, data_high, sd_low, data_low
        )

    print("max interface diffusive error:", diff.max())
    if current_max < diff.max():
        current_max = diff.max()
        id_max = intf.id