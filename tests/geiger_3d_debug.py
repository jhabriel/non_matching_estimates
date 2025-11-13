# import porepy as pp
# import numpy as np
#
# import mdnme
# from mdnme.estimates.diffusive_error import (
#     _interface_diffusive_error_1d_nonmatching,
#     _interface_diffusive_error_2d_nonmatching
# )
# from mdnme.utils.grid_rotation import build_canonical_frames, rotate_grid
# from mdnme.examples.bit_example_2.geometry import (
#     create_mdg_from_msh_file,
#     _paths_for_level,
#     _stem_for_refinement_level
# )
# from mdnme.utils.nested_refinement import GeoNestedRefinementFactory
#
# # Produce mixed-dimensional grid
# ref_lvl = 0
# non_matching = True
# geo_path, msh_path, csv_path, out_stem = _paths_for_level(ref_lvl)
# print('Generating nonmatching grid...')
#
# # Create two mixed-dimensional grids. The second is a nested refinement of the first
# factory = GeoNestedRefinementFactory(
#     src_path=str(geo_path),
#     dim=3,
#     num_refinements=1,
#     out_stem=out_stem,  # will emit <out_stem>_0.msh, <out_stem>_1.msh, ...
# )
#
# # Retrieve coarse and fine mixed-dimensional grids
# mdg_coarse = None
# mdg_fine = None
# for i, mdg in enumerate(factory):
#     if i == 0:
#         mdg_coarse = mdg
#     else:
#         mdg_fine = mdg
#
# # Create coarse-to-fine mapping
# sd_map = {}
# for sd_co, sd_fi in zip(mdg_coarse.subdomains(), mdg_fine.subdomains()):
#     assert sd_co.dim == sd_fi.dim
#     if 0 < sd_co.dim < 3:
#         sd_map[sd_co] = sd_fi
# # Replace lower-dimensional subdomains
# mdg_coarse.replace_subdomains_and_interfaces(sd_map=sd_map)
# mdg = mdg_coarse.copy()
#
# exporter = pp.Exporter(mdg, file_name="crossing", folder_name="crossing")
# exporter.write_vtu()
#
# # --- 1. Canonical frames for all grids & interfaces ---
#
# build_canonical_frames(mdg)
#
# # --- 2. Manufactured global linear pressure p(x,y,z) = alpha * z + c0 ---
#
# alpha = 1.0
# c0 = 0.3
#
# for sd, d in mdg.subdomains(return_data=True):
#     # ensure estimates dict
#     d.setdefault("estimates", {})
#
#     if sd.dim == 0:
#         # nothing to do here
#         continue
#
#     # physical 3D coordinates
#     X_phys = sd.nodes  # shape (3, n_nodes)
#
#     # canonically rotated subdomain grid
#     g_rot = rotate_grid(sd)
#     X_loc = g_rot.nodes    # shape (sd.dim, n_nodes) in canonical frame
#
#     cn = sd.cell_nodes().tocsc()
#
#     if sd.dim == 2:
#         # P1 coeffs [a_x, a_y, c] in the local 2D coords X_loc
#         C = np.empty((sd.num_cells, 3))
#         for K in range(sd.num_cells):
#             i0, i1 = cn.indptr[K], cn.indptr[K + 1]
#             nodes_K = cn.indices[i0:i1]
#             assert nodes_K.size == 3, "Simplex 2D grid expected."
#
#             xloc = X_loc[:, nodes_K]   # (2,3)
#             xphys = X_phys[:, nodes_K] # (3,3)
#
#             # exact values p(z) at the three vertices
#             z = xphys[2, :]            # z-coordinates
#             vals = alpha * z + c0      # (3,)
#
#             # solve [x y 1] * [a_x, a_y, c]^T = vals
#             V = np.vstack((xloc, np.ones(3)))  # (3,3)
#             C[K, :] = np.linalg.solve(V.T, vals)
#         d["estimates"]["recon_sd_pressure"] = C
#
#     elif sd.dim == 1:
#         # P1 coeffs [a_s, b] in the local 1D coordinate s = X_loc[0]
#         C = np.empty((sd.num_cells, 2))
#         s_all = X_loc[0, :]           # 1D coordinate in canonical frame
#
#         for K in range(sd.num_cells):
#             i0, i1 = cn.indptr[K], cn.indptr[K + 1]
#             nodes_K = cn.indices[i0:i1]
#             assert nodes_K.size == 2, "Simplex 1D grid expected."
#
#             n0, n1 = nodes_K
#             s0, s1 = s_all[n0], s_all[n1]
#             z0, z1 = X_phys[2, n0], X_phys[2, n1]
#
#             u0 = alpha * z0 + c0
#             u1 = alpha * z1 + c0
#
#             if abs(s1 - s0) < 1e-14:
#                 # degenerate, treat as constant
#                 a = 0.0
#                 b = u0
#             else:
#                 a = (u1 - u0) / (s1 - s0)
#                 b = u0 - a * s0
#             C[K, :] = [a, b]
#
#         d["estimates"]["recon_sd_pressure"] = C
#
#     else:
#         # 3D subdomains: not needed for the 1D interface test; skip or extend similarly
#         continue
#
# # --- 3. Zero flux and unit permeability on all interfaces ---
#
# for intf, d in mdg.interfaces(return_data=True):
#     d.setdefault("estimates", {})
#
#     # zero mortar flux
#     d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)
#
#     # unit normal permeability
#     d.setdefault(pp.PARAMETERS, {})
#     d[pp.PARAMETERS].setdefault("flow", {})
#     d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0
#
# # --- 4. Measure 1D nonmatching interface diffusive errors ---
#
# current_max = 0.0
# worst_intf_id = None
#
# for intf, data_intf in mdg.interfaces(return_data=True):
#     if intf.dim != 1:
#         continue
#
#     sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
#     data_high = mdg.subdomain_data(sd_high)
#     data_low = mdg.subdomain_data(sd_low)
#
#     diff = _interface_diffusive_error_1d_nonmatching(
#         intf, data_intf, sd_high, data_high, sd_low, data_low
#     )
#     print(f"Interface {intf.id}: max interface diffusive error = {diff.max()}")
#
#     if diff.max() > current_max:
#         current_max = diff.max()
#         worst_intf_id = intf.id
#
# print("Worst 1D interface error:", current_max, "at interface", worst_intf_id)
#
#
#

import numpy as np
import porepy as pp

import mdnme
from mdnme.estimates.diffusive_error import (
    _interface_diffusive_error_1d_nonmatching,
    _interface_diffusive_error_1d,
)
from mdnme.utils.grid_rotation import build_canonical_frames, rotate_grid
from mdnme.examples.bit_example_2.geometry import (
    _paths_for_level,
)
from mdnme.utils.nested_refinement import GeoNestedRefinementFactory


# -------------------------------------------------------------------
# 0. Build nested non-matching Geiger mdg (same pattern as your script)
# -------------------------------------------------------------------
ref_lvl = 0
geo_path, msh_path, csv_path, out_stem = _paths_for_level(ref_lvl)
print("Generating nonmatching grid...")

factory = GeoNestedRefinementFactory(
    src_path=str(geo_path),
    dim=3,
    num_refinements=1,
    out_stem=out_stem,
)

mdg_coarse = None
mdg_fine = None
for i, mdg_tmp in enumerate(factory):
    if i == 0:
        mdg_coarse = mdg_tmp
    else:
        mdg_fine = mdg_tmp

# map 1D/2D coarse → fine (nested refinement) and replace on coarse mdg
sd_map = {}
for sd_co, sd_fi in zip(mdg_coarse.subdomains(), mdg_fine.subdomains()):
    assert sd_co.dim == sd_fi.dim
    if 2 < sd_co.dim < 3:
        sd_map[sd_co] = sd_fi

mdg_coarse.replace_subdomains_and_interfaces(sd_map=sd_map)
mdg = mdg_coarse.copy()

print("Done generating nested nonmatching mdg.")

# Optional: export to VTK for inspection
# exporter = pp.Exporter(mdg, file_name="geiger_random_p1", folder_name="geiger_random_p1")
# exporter.write_vtu()

# -------------------------------------------------------------------
# 1. Canonical frames for all grids & interfaces
# -------------------------------------------------------------------
build_canonical_frames(mdg)

# -------------------------------------------------------------------
# 2. Build a random P1-conforming field on 1D and 2D subdomains
#
#    Idea:
#      - draw random cell values u_K
#      - nodal value = average of incident cell values
#      - reconstruct cellwise P1 in canonical coordinates
# -------------------------------------------------------------------
rng = np.random.default_rng(seed=12345)

for sd, d in mdg.subdomains(return_data=True):
    d.setdefault("estimates", {})

    if sd.dim == 0 or sd.dim == 3:
        # we skip 0D and 3D for this test; only 1D and 2D matter for 1D interfaces
        continue

    # cell-node incidence
    cn = sd.cell_nodes().tocsc()

    # 1) random cell values
    u_cell = rng.standard_normal(sd.num_cells)

    # 2) accumulate to nodes: u_node = average of incident cell values
    u_node = np.zeros(sd.num_nodes)
    counts = np.zeros(sd.num_nodes, dtype=int)
    for K in range(sd.num_cells):
        i0, i1 = cn.indptr[K], cn.indptr[K + 1]
        nodes_K = cn.indices[i0:i1]
        u_node[nodes_K] += u_cell[K]
        counts[nodes_K] += 1
    # avoid division by zero (shouldn't happen in a proper grid)
    mask = counts > 0
    u_node[mask] /= counts[mask]

    # 3) reconstruct cellwise P1 coefficients in the canonical frame
    g_rot = rotate_grid(sd)
    X_loc = g_rot.nodes[: sd.dim, :]  # (dim, n_nodes)

    if sd.dim == 2:
        # C[K, :] = [a_x, a_y, c], u(x,y) = a_x x + a_y y + c
        C = np.empty((sd.num_cells, 3))
        for K in range(sd.num_cells):
            i0, i1 = cn.indptr[K], cn.indptr[K + 1]
            nodes_K = cn.indices[i0:i1]
            assert nodes_K.size == 3, "Expected simplex 2D grid (3 nodes per cell)."

            xy = X_loc[:, nodes_K]         # (2, 3)
            vals = u_node[nodes_K]         # (3,)

            V = np.vstack((xy, np.ones(3)))  # 3x3: rows [x; y; 1]
            C[K, :] = np.linalg.solve(V.T, vals)
        d["estimates"]["recon_sd_pressure"] = C

    elif sd.dim == 1:
        # C[K, :] = [a_s, b], u(s) = a_s * s + b
        C = np.empty((sd.num_cells, 2))
        s_all = X_loc[0, :]  # 1D coordinate in canonical frame

        for K in range(sd.num_cells):
            i0, i1 = cn.indptr[K], cn.indptr[K + 1]
            nodes_K = cn.indices[i0:i1]
            assert nodes_K.size == 2, "Expected simplex 1D grid (2 nodes per cell)."

            n0, n1 = nodes_K
            s0, s1 = s_all[n0], s_all[n1]
            u0, u1 = u_node[n0], u_node[n1]

            if abs(s1 - s0) < 1e-14:
                # degenerate: treat as constant
                a = 0.0
                b = u0
            else:
                a = (u1 - u0) / (s1 - s0)
                b = u0 - a * s0

            C[K, :] = [a, b]

        d["estimates"]["recon_sd_pressure"] = C

print("Random P1-conforming recon_sd_pressure assigned on 1D and 2D subdomains.")

# -------------------------------------------------------------------
# 3. Zero flux and unit permeability on all interfaces
# -------------------------------------------------------------------
for intf, d in mdg.interfaces(return_data=True):
    d.setdefault("estimates", {})
    d["estimates"]["fv_intf_flux"] = np.zeros(intf.num_cells)

    d.setdefault(pp.PARAMETERS, {})
    d[pp.PARAMETERS].setdefault("flow", {})
    d[pp.PARAMETERS]["flow"]["effective_permeability"] = 1.0

print("Interface data (zero flux, unit permeability) set.")

# -------------------------------------------------------------------
# 4. Measure 1D nonmatching interface diffusive errors
#    and highlight interfaces with 4 cells
# -------------------------------------------------------------------
current_max = 0.0
worst_intf_id = None

#print("\n1D interface diffusive errors (random P1 field):")
#print(" id   dim  n_cells    max|diff|   (mark 4-cell intfs)")
#print("-----------------------------------------------------")

for intf, data_intf in mdg.interfaces(return_data=True):
    if intf.dim != 1:
        continue

    sd_high, sd_low = mdg.interface_to_subdomain_pair(intf)
    data_high = mdg.subdomain_data(sd_high)
    data_low = mdg.subdomain_data(sd_low)

    diff = _interface_diffusive_error_1d_nonmatching(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )

    _interface_diffusive_error_1d(
        intf, data_intf, sd_high, data_high, sd_low, data_low
    )


#     max_diff = float(np.max(np.abs(diff)))
#     mark = " <== 4 cells" if intf.num_cells == 4 else ""
#     print(f"{intf.id:4d}   {intf.dim:d}    {intf.num_cells:4d}   {max_diff:10.3e}{mark}")
#
#     if max_diff > current_max:
#         current_max = max_diff
#         worst_intf_id = intf.id
#
# print("\nWorst 1D interface diffusive error:")
# print(f"Interface {worst_intf_id} with max|diff| = {current_max:.3e}")
