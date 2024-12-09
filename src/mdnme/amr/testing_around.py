import porepy as pp
import numpy as np
import scipy.sparse as sps

# # %% Create a triangular grid and convert it to a FEM mesh, suitable to employ it as
# # part of an adaptive mesh refinement strategy
# nx = np.array([1, 1])
# physdims = np.ones(2)
# g = pp.StructuredTriangleGrid(nx, physdims)
#
# coordinates = g.nodes.transpose()[:, :g.dim]
# elements = g.cell_node_matrix()


# Define a rectangular domain in terms of range in the two dimensions
bounding_box = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
domain = pp.Domain(bounding_box=bounding_box)

# Define each individual fracture, collect into a list.
frac = pp.LineFracture(np.array([[0.5, 0.5], [0, 1]]))
fractures = [frac]

# Define a fracture network in 2d
network_2d = pp.create_fracture_network(fractures, domain)

# Set overall target cell size and target cell size close to the fracture.
mesh_args: dict[str, float] = {"cell_size": 0.5, "cell_size_fracture": 0.5}

# Generate a mixed-dimensional grid
mdg = pp.create_mdg("simplex", mesh_args, network_2d)

pp.save_img("test_grid", mdg, alpha=0.25, plot_2d=True)
