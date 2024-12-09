import porepy as pp
import numpy as np

# --> Testing `replace_grid` with one vertical fracture line embedded in a 2d matrix

# Geometric setup
domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})
line_fracture = pp.LineFracture(np.array([[0.5, 0.5], [0.25, 0.75]]))
fn = pp.create_fracture_network([line_fracture], domain)

# Create base mdg and extract subdomains and interface
mdg_base = pp.create_mdg("simplex", {"cell_size": 0.25}, fn)
sd_matrix_base = mdg_base.subdomains()[0]
sd_fracture_base = mdg_base.subdomains()[1]
intf_base = mdg_base.interfaces()[0]

# Create refine mdg and extract subdomains and interface
mdg_fine = pp.create_mdg("cartesian", {"cell_size": 0.05}, fn)
sd_matrix_fine = mdg_fine.subdomains()[0]
sd_fracture_fine = mdg_fine.subdomains()[1]
intf_fine = mdg_fine.interfaces()[0]

# Test 1: Replace only the matrix grid
mdg_new_matrix_replaced = mdg_base.copy()
mdg_new_matrix_replaced.replace_subdomains_and_interfaces(
        sd_map={sd_matrix_base: sd_matrix_fine}
)
print("TEST 1")
print(mdg_new_matrix_replaced)
print("OK")
print(100*"-")

# Test 2: Replace only the fracture grid
mdg_new_frac_replaced = mdg_base.copy()
mdg_new_frac_replaced.replace_subdomains_and_interfaces(
    sd_map={sd_fracture_base: sd_fracture_fine}
)
print("TEST 3")
print(mdg_new_frac_replaced)
print("OK")
print(100*"-")

# Test 3: Replace only the interface grid
mdg_new_intf_replaced = mdg_base.copy()
mdg_new_intf_replaced.replace_subdomains_and_interfaces(
    intf_map={intf_base: intf_fine}
)
print("TEST 2")
print(mdg_new_intf_replaced)
print("OK")
print(100*"-")



#
# # In the following, mdg_coarse is mdg_old
# mdg_new_change_matrix = mdg_coarse.copy()
# mdg_new_change_matrix.replace_subdomains_and_interfaces(
#     sd_map={sd_matrix_coarse: sd_matrix_fine}
# )
# # pp.save_img("mdg_replaced_0.png", mdg_new, plot_2d=True)
#
# mdg_new_change_fracture = mdg_coarse.copy()
# mdg_new_change_fracture.replace_subdomains_and_interfaces(
#     sd_map={sd_fracture_coarse: sd_fracture_fine}
# )
#
# mdg_new_change_mortar = mdg_coarse.copy()
# mdg_new_change_mortar.replace_subdomains_and_interfaces(
#     intf_map={intf_coarse: intf_fine}
# )
#
# # pp.save_img("mdg_replaced_0.png", mdg_new, plot_2d=True)
#
#
# # # Create a two-dimensional simplicial grid
# # sd_triangle = pp.StructuredTriangleGrid(nx=np.array([4, 4]), physdims=np.array([1, 1]))
# # sd_triangle.compute_geometry()
# # # pp.save_img("triangle_structured.png", sd_triangle)
# # mdg_new = mdg_coarse.copy()
# # mdg_new.replace_subdomains_and_interfaces(
# #     sd_map={mdg_new.subdomains()[0]: sd_triangle}
# # )