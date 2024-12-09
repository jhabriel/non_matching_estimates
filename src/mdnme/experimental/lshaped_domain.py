"""
Adaptive mesh refinement for an L-shaped domain with.

Boundary conditions:
    Left: constant pressure = 1
    Right: constant pressure = 0
    Elsewhere: no flux

"""

# STEP 0 -> Import libraries
import numpy as np
import scipy.sparse as sps
import porepy as pp
import mdnme

from mdnme.estimates.error_estimation import estimate_errors

from mdnme.amr.refinement_utils import plot_fem_mesh
from mdnme.porepy_interface.porepy_grid_to_fem_grid import porepy_grid_to_fem_grid
from mdnme.porepy_interface.fem_grid_to_porepy_grid import fem_grid_to_sd_grid_2d

# STEP 1 -> Create initial grid

# Create initial grid
coordinates = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
    ]
)

elements = np.array(
    [
        [0, 1, 3],
        [1, 4, 3],
        [1, 2, 4],
        [2, 5, 4],
        [3, 4, 6],
        [4, 7, 6],
    ]
)

# Uncomment next line to plot the original FEM mesh
# plot_fem_mesh(coordinates, elements)

# Convert to a PorePy grid
g = fem_grid_to_sd_grid_2d(coordinates, elements)

# Uncomment next line to plot the original PorePy grid
# pp.save_img("base_grid.png", info="nf", grid=g, plot_2d=True)

# STEP 2 -> Define model
# Permeability
perm = pp.SecondOrderTensor(np.ones(g.num_cells))

# Boundary conditions
b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
fc = g.face_centers
dir_faces = np.where((fc[0] < 1e-6) + (fc[0] > (2 - 1e-6)))[0]
bc = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
bc_val = np.zeros(g.num_faces)
bc_val[g.face_centers[0] < 1e-6] = 1.0

# Collect all parameters in a dictionary
parameters = {"second_order_tensor": perm, "bc": bc, "bc_values": bc_val}
data_key = "flow"
data = pp.initialize_default_data(g, {}, data_key, parameters)

# Solve
flow_discretization = pp.Mpfa(data_key)
flow_discretization.discretize(g, data)
A, b_flow = flow_discretization.assemble_matrix_rhs(g, data)
p_mpfa = sps.linalg.spsolve(A, b_flow)

# Compute Darcy flux
mpfa_flux = data[pp.DISCRETIZATION_MATRICES][data_key]["flux"]
mpfa_bc_flux = data[pp.DISCRETIZATION_MATRICES][data_key]["bound_flux"]
darcy_flux = mpfa_flux * p_mpfa + mpfa_bc_flux * bc_val

data["estimates"] = {}
data["estimates"]["fv_sd_pressure"] = p_mpfa
data["estimates"]["fv_sd_flux"] = darcy_flux

# Create empty mixed-dimensional grid
mdg = pp.MixedDimensionalGrid()
mdg.add_subdomains([g])
mdg._subdomain_data = {g: data}

# Uncomment next line to plot solution
# pp.save_img("pressure_solution.png", grid=g, cell_value=p_tpfa, plot_2d=True)

# STEP 3 -> estimate error

# Estimate error
estimate_errors(
    mdg=mdg,
    pressure_reconstruction_method="patchwise_p1",
    sources=None,
    quadrature_degree_for_residual_error=None
)
