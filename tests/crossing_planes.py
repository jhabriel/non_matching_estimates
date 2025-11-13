import numpy as np
import porepy as pp

from porepy.grids.refinement import GridSequenceFactory

import mdnme
from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_1d_nonmatching,
    _interface_diffusive_error_2d_nonmatching, _interface_diffusive_error_1d
)
from mdnme.utils.grid_rotation import build_canonical_frames, rotate_grid

# Create a 3d mdg with two intersecting planes, and then produce a non-matching grid

domain = pp.Domain(
    {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 1, "ymax": 1, "zmax": 1}
)
f1 = pp.PlaneFracture(np.array([
    [0.5, 0.5, 0.5, 0.5],
    [0.25, 0.25, 0.75, 0.75],
    [0.25, 0.75, 0.75, 0.25],
]))

f2 = pp.PlaneFracture(np.array([
    [0.25, 0.25, 0.75, 0.75],
    [0.5, 0.5, 0.5, 0.5],
    [0.25, 0.75, 0.75, 0.25],
]))

# New horizontal fracture: z = 0.4
# f3 = pp.PlaneFracture(np.array([
#     [0.25, 0.25, 0.75, 0.75],  # x
#     [0.25, 0.75, 0.75, 0.25],  # y
#     [0.4 , 0.4 , 0.4 , 0.4 ],  # z
# ]))

#fl = [f1, f2, f3]
fl = [f1, f2]
fn = pp.create_fracture_network(fl, domain)

mdg_coarse = pp.create_mdg(
    "simplex",
    meshing_args={"cell_size": 0.2},
    fracture_network=fn
)

mdg_fine = pp.create_mdg(
    "simplex",
    meshing_args={"cell_size": 0.05},
    fracture_network=fn
)

# Establish mapping
sd_map = {}
for sd_coarse, sd_fine in zip(mdg_coarse.subdomains(), mdg_fine.subdomains()):
    if sd_coarse.dim < 3 and sd_coarse.dim > 2:
        sd_map[sd_coarse] = sd_fine

# Generate non-matching grid
mdg_coarse.replace_subdomains_and_interfaces(sd_map=sd_map)
mdg = mdg_coarse.copy()

exporter = pp.Exporter(mdg, file_name="crossing", folder_name="crossing")
exporter.write_vtu()


# --- 1. Canonical frames for all grids & interfaces ---

build_canonical_frames(mdg)

# --- 2. Manufactured global linear pressure p(x,y,z) = alpha * z + c0 ---

alpha = 1.0
c0 = 0.3

for sd, d in mdg.subdomains(return_data=True):
    # ensure estimates dict
    d.setdefault("estimates", {})

    if sd.dim == 0:
        # nothing to do here
        continue

    # physical 3D coordinates
    X_phys = sd.nodes  # shape (3, n_nodes)

    # canonically rotated subdomain grid
    g_rot = rotate_grid(sd)
    X_loc = g_rot.nodes    # shape (sd.dim, n_nodes) in canonical frame

    cn = sd.cell_nodes().tocsc()

    if sd.dim == 2:
        # P1 coeffs [a_x, a_y, c] in the local 2D coords X_loc
        C = np.empty((sd.num_cells, 3))
        for K in range(sd.num_cells):
            i0, i1 = cn.indptr[K], cn.indptr[K + 1]
            nodes_K = cn.indices[i0:i1]
            assert nodes_K.size == 3, "Simplex 2D grid expected."

            xloc = X_loc[:, nodes_K]   # (2,3)
            xphys = X_phys[:, nodes_K] # (3,3)

            # exact values p(z) at the three vertices
            z = xphys[2, :]            # z-coordinates
            vals = alpha * z + c0      # (3,)

            # solve [x y 1] * [a_x, a_y, c]^T = vals
            V = np.vstack((xloc, np.ones(3)))  # (3,3)
            C[K, :] = np.linalg.solve(V.T, vals)
        d["estimates"]["recon_sd_pressure"] = C

    elif sd.dim == 1:
        # P1 coeffs [a_s, b] in the local 1D coordinate s = X_loc[0]
        C = np.empty((sd.num_cells, 2))
        s_all = X_loc[0, :]           # 1D coordinate in canonical frame

        for K in range(sd.num_cells):
            i0, i1 = cn.indptr[K], cn.indptr[K + 1]
            nodes_K = cn.indices[i0:i1]
            assert nodes_K.size == 2, "Simplex 1D grid expected."

            n0, n1 = nodes_K
            s0, s1 = s_all[n0], s_all[n1]
            z0, z1 = X_phys[2, n0], X_phys[2, n1]

            u0 = alpha * z0 + c0
            u1 = alpha * z1 + c0

            if abs(s1 - s0) < 1e-14:
                # degenerate, treat as constant
                a = 0.0
                b = u0
            else:
                a = (u1 - u0) / (s1 - s0)
                b = u0 - a * s0
            C[K, :] = [a, b]

        d["estimates"]["recon_sd_pressure"] = C

    else:
        # 3D subdomains: not needed for the 1D interface test; skip or extend similarly
        continue

# --- 3. Zero flux and unit permeability on all interfaces ---

for intf, d in mdg.interfaces(return_data=True):
    d.setdefault("estimates", {})

    # zero mortar flux
    d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)

    # unit normal permeability
    d.setdefault(pp.PARAMETERS, {})
    d[pp.PARAMETERS].setdefault("flow", {})
    d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0

# --- 4. Measure 1D nonmatching interface diffusive errors ---

current_max = 0.0
worst_intf_id = None

for intf, data_intf in mdg.interfaces(return_data=True):
    if intf.dim != 1:
        continue

    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low = mdg.subdomain_data(sd_low)

    diff = _interface_diffusive_error_1d(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )

    diff = _interface_diffusive_error_1d_nonmatching(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )
    #print(f"Interface {intf.id}: max interface diffusive error = {diff.max()}")

#     if diff.max() > current_max:
#         current_max = diff.max()
#         worst_intf_id = intf.id
#
# print("Worst 1D interface error:", current_max, "at interface", worst_intf_id)