import numpy as np

import mdnme
import porepy as pp
import scipy.sparse as sps

from mdnme.porepy_interface.porepy_grid_to_fem_grid import porepy_grid_to_fem_grid
from mdnme.porepy_interface.fem_grid_to_porepy_grid import fem_grid_to_sd_grid_2d
from mdnme.amr.refinement_utils import plot_fem_mesh, random_marking

# Create a unit square mesh with two triangles
nx = np.array([12, 5])
physdims = np.ones(2)
sd = pp.StructuredTriangleGrid(nx, physdims)
sd.compute_geometry()
pp.save_img("original_grid", sd, alpha=0.1, plot_2d=True)


# Convert PorePy grid into a FEM grid
coo, ele = porepy_grid_to_fem_grid(g=sd)

# Define elements to be refined
marked = random_marking(ele)

new_coo, new_ele, _, _, _ = mdnme.refine_nvb(coo, ele, marked)

plot_fem_mesh(new_coo, new_ele)

# Convert the refine grid into a PorePy grid
g_refined = fem_grid_to_sd_grid_2d(new_coo, new_ele)
pp.save_img("refined_grid", g_refined, alpha=0.1, plot_2d=True)
