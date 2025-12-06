"""
Module containing the mixin class to generate the mixed-dimensional grid associated
to the geometry used in the manufactured solution from Appendix D.2. from [1].

Reference:
    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. & Radu, F. (2023). A
      posteriori error estimates for hierarchical mixed-dimensional elliptic
      equations. Journal of Numerical Mathematics, 31(4), 247-280.
      https://doi.org/10.1515/jnma-2022-0038

"""

from typing import Callable, Literal

import numpy as np
import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.grids.refinement import GridSequenceFactory

from mdnme.utils.grid_rotation import build_canonical_frames


class VarelaJNumGeometry3D:
    """Generate fracture network and mixed-dimensional grid."""

    params: dict
    """Simulation model parameters."""

    grid_type: Callable[[], Literal["cartesian", "simplex", "tensor_grid"]]
    """Type of grid."""

    def set_fractures(self) -> None:
        """Declare set of fractures.

        Note:
            The physical fracture is `physical_frac_0`. For simplices, fractures
            `ghost_frac_0` ... `ghost_frac_23` are constraints for the meshing process.

        """

        physical_frac_0 = pp.PlaneFracture(
            np.array(
                [
                    [0.50, 0.50, 0.50, 0.50],
                    [0.25, 0.25, 0.75, 0.75],
                    [0.25, 0.75, 0.75, 0.25],
                ]
            )
        )

        if self.grid_type() == "simplex":
            ghost_frac_0 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_1 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_2 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_3 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.25, 0.25],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_4 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.50, 0.00, 0.00],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_5 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 1.00, 0.50, 0.50],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_6 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.50, 0.00, 0.00],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_7 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 1.00, 0.50, 0.50],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.75, 0.75, 0.25],
                    ]
                )
            )
            ghost_frac_8 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_9 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_10 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_11 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 1.00, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_12 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_13 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.25, 0.25, 0.25, 0.25],
                    ]
                )
            )
            ghost_frac_14 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_15 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.00, 0.00],
                        [0.75, 0.75, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_16 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_17 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_18 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.75, 0.75, 0.75, 0.75],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_19 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.75, 0.75, 0.75, 0.75],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_20 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_21 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.00, 0.00],
                    ]
                )
            )
            ghost_frac_22 = pp.PlaneFracture(
                np.array(
                    [
                        [0.50, 0.00, 0.00, 0.50],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )
            ghost_frac_23 = pp.PlaneFracture(
                np.array(
                    [
                        [1.00, 0.50, 0.50, 1.00],
                        [0.25, 0.25, 0.25, 0.25],
                        [1.00, 1.00, 0.75, 0.75],
                    ]
                )
            )

            self._fractures = [
                physical_frac_0,
                ghost_frac_0,
                ghost_frac_1,
                ghost_frac_2,
                ghost_frac_3,
                ghost_frac_4,
                ghost_frac_5,
                ghost_frac_6,
                ghost_frac_7,
                ghost_frac_8,
                ghost_frac_9,
                ghost_frac_10,
                ghost_frac_11,
                ghost_frac_12,
                ghost_frac_13,
                ghost_frac_14,
                ghost_frac_15,
                ghost_frac_16,
                ghost_frac_17,
                ghost_frac_18,
                ghost_frac_19,
                ghost_frac_20,
                ghost_frac_21,
                ghost_frac_22,
                ghost_frac_23,
            ]

        elif self.grid_type() == "cartesian":
            self._fractures = [physical_frac_0]
        else:
            raise NotImplementedError()

    def set_domain(self) -> None:
        """Set domain"""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Define mesh arguments for meshing."""
        return self.params.get("meshing_arguments", {"cell_size": 0.125})

    def meshing_kwargs(self) -> dict:
        """Declare meshing constraints. Ignore ghost fractures 1 to 24."""
        kw_args = {}
        if self.grid_type() == "simplex":
            kw_args.update({"constraints": np.arange(1, 25)})
        return kw_args

    def set_geometry(self) -> None:

        # Create the geometry through domain and fracture set.
        self.set_domain()
        self.set_fractures()

        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(
            self.fractures,
            self.domain,
        )
        # Check if we should build a matching or a non-matching mdg
        is_nonmatching = bool(self.params.get("non_matching", False))

        if is_nonmatching:

            # Non-matching requested: choose mechanism
            perturb_frac = self.params.get("perturb_fracture", False)
            perturb_mortar = self.params.get("perturb_mortar", False)
            refine_fracture = self.params.get("refine_fracture", False)
            refine_mortar = self.params.get("refine_mortar", False)

            # Sanity: at least one mechanism should be specified
            if not (perturb_frac or perturb_mortar or refine_fracture or refine_mortar):
                raise ValueError(
                    "non_matching=True but no perturbation or refine option was "
                    "defined. Set one of {'perturb_fracture', 'perturb_mortar', "
                    "'refine_fracture', 'refine_mortar'} in params."
                )

            # >>> Perturbation-based nonmatchingness <<<
            if perturb_frac or perturb_mortar:

                # NOTE: Do not put this outside the if-statement or things crash!
                # Start from a matching mdg (will be overwritten below)
                mdg_final = pp.create_mdg(
                    grid_type=self.grid_type(),
                    meshing_args=self.meshing_arguments(),
                    fracture_network=self.fracture_network,
                    **self.meshing_kwargs(),
                )

                # Sanity check on translation vector
                if self.params.get("translation_vector") is None:
                    raise ValueError("Expected a translation vector")
                tvec = self.params.get("translation_vector")
                x_move, y_move, z_move = tvec[0], tvec[1], tvec[2]
                if not np.isclose(x_move, 0):
                    raise ValueError("Points cannot be moved in the x-direction")
                if np.isclose(y_move, 0) and np.isclose(z_move, 0):
                    raise ValueError("Expected translation in the y or z direction")

                # We only perturb the fracture grid
                if perturb_frac and not perturb_mortar:

                    # Retrieve fracture grid
                    frac_grid = mdg_final.subdomains(dim=2)[0]
                    pert_frac_grid = frac_grid.copy()

                    # Get amplitude of translation. If not given, we use half of the
                    # mean cell diameter of the grid
                    default_amp = frac_grid.cell_diameters().mean() / 2
                    amp: float = self.params.get("amplitude", default_amp)

                    # Retrieve "boundary" nodes
                    y_nodes = pert_frac_grid.nodes[1]
                    z_nodes = pert_frac_grid.nodes[2]
                    bound_nodes_mask = (
                        np.isclose(y_nodes, 0.25)
                        + np.isclose(y_nodes, 0.75)
                        + np.isclose(z_nodes, 0.25)
                        + np.isclose(z_nodes, 0.75)
                    )
                    int_nodes_mask = np.logical_not(bound_nodes_mask)

                    # Translate all internal nodes by a constant value `amp`
                    # Direction of motion is determined by the translation vector
                    pert_frac_grid.nodes[1][int_nodes_mask] += y_move * amp
                    pert_frac_grid.nodes[2][int_nodes_mask] += z_move * amp
                    pert_frac_grid.compute_geometry()

                    # Now, we have to replace the fracture grid into the mdg....
                    mdg_final.replace_subdomains_and_interfaces(
                        sd_map={frac_grid: pert_frac_grid}
                    )

                # We only perturb the mortar grid
                if perturb_mortar and not perturb_frac:

                    # Retrieve interface grid
                    intf = mdg_final.interfaces(dim=2)[0]

                    sg_map: dict = {}
                    # Loop over the two sides of the mortar grid
                    for proj_msg, mg_side in intf.project_to_side_grids():

                        # Identify the side enum that owns this mortar side grid
                        side_enum = next(
                            k for k, v in intf.side_grids.items() if v is mg_side
                        )

                        # Make a hard copy of the sidegrid instead
                        pert_mg_side = mg_side.copy()

                        # Get amplitude of translation. If not given, we use half of the
                        # mean cell diameter of the grid
                        default_amp = mg_side.cell_diameters().mean() / 2
                        amp: float = self.params.get("amplitude", default_amp)

                        # Retrieve "boundary" nodes
                        y_nodes = pert_mg_side.nodes[1]
                        z_nodes = pert_mg_side.nodes[2]
                        bound_nodes_mask = (
                            np.isclose(y_nodes, 0.25)
                            + np.isclose(y_nodes, 0.75)
                            + np.isclose(z_nodes, 0.25)
                            + np.isclose(z_nodes, 0.75)
                        )
                        int_nodes_mask = np.logical_not(bound_nodes_mask)

                        # Translate all internal nodes by a constant value `amp`
                        # Direction of motion is determined by the translation vector
                        pert_mg_side.nodes[1][int_nodes_mask] += y_move * amp
                        pert_mg_side.nodes[2][int_nodes_mask] += z_move * amp
                        pert_mg_side.compute_geometry()

                        # Store in map
                        sg_map[side_enum] = pert_mg_side

                    # Finally, perform the replacement
                    mdg_final.replace_subdomains_and_interfaces(
                        interface_map={intf: sg_map}
                    )

                # We perturb both the fracture and the mortar grid
                if perturb_frac and perturb_mortar:

                    # 1: Perturb fracture grid

                    # Retrieve fracture grid
                    frac_grid = mdg_final.subdomains(dim=2)[0]
                    pert_frac_grid = frac_grid.copy()

                    # Get amplitude of translation. If not given, we use half of the
                    # mean cell diameter of the grid
                    default_amp_frac = frac_grid.cell_diameters().mean() / 2
                    amp_frac: float = self.params.get("amplitude", default_amp_frac)

                    # Retrieve "boundary" nodes
                    y_nodes_frac = pert_frac_grid.nodes[1]
                    z_nodes_frac = pert_frac_grid.nodes[2]
                    bound_nodes_mask_frac = (
                        np.isclose(y_nodes_frac, 0.25)
                        + np.isclose(y_nodes_frac, 0.75)
                        + np.isclose(z_nodes_frac, 0.25)
                        + np.isclose(z_nodes_frac, 0.75)
                    )
                    int_nodes_mask_frac = np.logical_not(bound_nodes_mask_frac)

                    # Translate all internal nodes by a constant value `amp`
                    # Direction of motion is determined by the translation vector
                    pert_frac_grid.nodes[1][int_nodes_mask_frac] += y_move * amp_frac
                    pert_frac_grid.nodes[2][int_nodes_mask_frac] += z_move * amp_frac
                    pert_frac_grid.compute_geometry()

                    # 2: Perturb sidegrids of the interface grid

                    # Retrieve interface grid
                    intf = mdg_final.interfaces(dim=2)[0]

                    sg_map: dict = {}
                    # Loop over the two sides of the mortar grid
                    for proj_msg, mg_side in intf.project_to_side_grids():

                        # Identify the side enum that owns this mortar side grid
                        side_enum = next(
                            k for k, v in intf.side_grids.items() if v is mg_side
                        )

                        # Make a hard copy of the sidegrid instead
                        pert_mg_side = mg_side.copy()

                        # Get amplitude of translation. If not given, we use half of the
                        # mean cell diameter of the grid
                        default_amp_mortar = mg_side.cell_diameters().mean() / 2
                        amp_mortar: float = self.params.get(
                            "amplitude", default_amp_mortar
                        )

                        # Retrieve "boundary" nodes
                        y_nodes_mortar = pert_mg_side.nodes[1]
                        z_nodes_mortar = pert_mg_side.nodes[2]
                        bound_nodes_mask_mortar = (
                            np.isclose(y_nodes_mortar, 0.25)
                            + np.isclose(y_nodes_mortar, 0.75)
                            + np.isclose(z_nodes_mortar, 0.25)
                            + np.isclose(z_nodes_mortar, 0.75)
                        )
                        int_nodes_mask_mortar = np.logical_not(bound_nodes_mask_mortar)

                        # Translate all internal nodes by a constant value `amp`
                        # Direction of motion is determined by the translation vector
                        # Note: We translate the nodes in the opposite side of the
                        # translation vector when both fracture and mortar grid
                        # internal nodes are perturbed
                        pert_mg_side.nodes[1][int_nodes_mask_mortar] += (
                            -y_move * amp_mortar
                        )
                        pert_mg_side.nodes[2][int_nodes_mask_mortar] += (
                            -z_move * amp_mortar
                        )
                        pert_mg_side.compute_geometry()

                        # Store in map
                        sg_map[side_enum] = pert_mg_side

                    # 3: Replace fracture grid and mortar grid
                    mdg_final.replace_subdomains_and_interfaces(
                        sd_map={frac_grid: pert_frac_grid}, interface_map={intf: sg_map}
                    )

            # >>> Refinement-based non-matchingness <<<
            if refine_fracture or refine_mortar:

                if perturb_frac or perturb_mortar:
                    raise ValueError("Cannot perturb and refine simultaneously.")

                # Obtain GridSequenceFactory parameters from standard API
                mesh_size_bound = self.params["meshing_arguments"]["cell_size"]
                mesh_size_frac = self.params["meshing_arguments"].get(
                    "cell_size_fracture", mesh_size_bound
                )
                mesh_size_min = self.params["meshing_arguments"].get(
                    "cell_size_min", 0.05 * mesh_size_bound
                )

                grid_sequence_params = {
                    "mode": "nested",
                    "num_refinements": 2,
                    "mesh_param": {
                        "mesh_size_bound": mesh_size_bound,
                        "mesh_size_frac": mesh_size_frac,
                        "mesh_size_min": mesh_size_min,
                    },
                    "grid_param": self.meshing_kwargs(),
                }

                factory = GridSequenceFactory(
                    self.fracture_network,
                    grid_sequence_params,
                )
                mdgs = list(factory)
                mdg_final = mdgs[0]
                mdg_fine = mdgs[1]

                # Replace grids
                if refine_fracture and not refine_mortar:
                    mdg_final.replace_subdomains_and_interfaces(
                        sd_map={
                            mdg_final.subdomains(dim=2)[0]: mdg_fine.subdomains(dim=2)[
                                0
                            ]
                        }
                    )
                elif refine_mortar and not refine_fracture:
                    mdg_final.replace_subdomains_and_interfaces(
                        interface_map={
                            mdg_final.interfaces(dim=2)[0]: mdg_fine.interfaces(dim=2)[
                                0
                            ]
                        }
                    )
                else:
                    mdg_final.replace_subdomains_and_interfaces(
                        sd_map={
                            mdg_final.subdomains(dim=2)[0]: mdg_fine.subdomains(dim=2)[
                                0
                            ]
                        },
                        interface_map={
                            mdg_final.interfaces(dim=2)[0]: mdg_fine.interfaces(dim=2)[
                                0
                            ]
                        },
                    )

                # Make sure to recompute the geometry of boundary grids
                for sd in mdg_final.subdomains():
                    bg = mdg_final.subdomain_to_boundary_grid(sd)
                    bg.compute_geometry()

        else:
            # The mdg is matching, and we create the mdg in the usual way
            mdg_final = pp.create_mdg(
                self.grid_type(),
                self.meshing_arguments(),
                self.fracture_network,
                **self.meshing_kwargs(),
            )

        # Finally, we have our mdg
        self.mdg: pp.MixedDimensionalGrid = mdg_final.copy()

        # Dimensionality of highest-dimensional manifold
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)
        build_canonical_frames(self.mdg)
