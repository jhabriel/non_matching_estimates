"""
Module containing a class that containing functionality to compute the true errors,
which are needed to compute the efficiency indices for the first numerical example of
the paper.

"""
import numpy as np
import porepy as pp
import quadpy
import scipy.sparse as sps
import sympy as sym

import mdnme
from mdnme.examples.varela_jnum_3d.exact_solution import VarelaJNumExactSolution3D


class VarelaJNumTrueErrors3D(VarelaJNumExactSolution3D):
    """
    Class containing the computation of true errors for the 3D manufactured solution.
    """

    def __init__(self, model):
        super().__init__(model)

    def true_error(self) -> float:
        """
        Compute global true error.

        Parameters:
            mdg: pp.MixedDimensionalGrid
                Mixed-dimensional grid of the problem.

        Returns:
            Value of the true error for the whole mixed-dimensional grid.

        """
        mdg = self.model.mdg
        sd_2d, d_2d = mdg.subdomains(return_data=True, dim=2)[0]
        sd_1d, d_1d = mdg.subdomains(return_data=True, dim=1)[0]
        intf, d_intf = mdg.interfaces(return_data=True)[0]

        te_sq_2d = self.true_error_matrix(sd_2d, d_2d)
        te_sq_1d = self.true_error_fracture(sd_1d, d_1d)
        te_sq_intf = self.true_error_interface(
            intf=intf,
            data_intf=d_intf,
            sd_high=sd_2d,
            data_high=d_2d,
            sd_low=sd_1d,
            data_low=d_1d,
        )

        true_error = (te_sq_2d.sum() + te_sq_1d.sum() + te_sq_intf.sum()) ** 0.5

        return true_error

    def true_error_matrix(self) -> np.ndarray:
        """
        Compute true error contribution of the matrix subdomain.

        Returns:
            True error contribution of the matrix, containing the local error squared
            of each element of the grid. Shape is `sd.num_cells`.

         Note:
             - We assume the field ``d_matrix["estimates"]["recon_sd_pressure"]`` is
               available.
             - We employ a numerical quadrature scheme that is accurate up to a
               polynomial degree 10.

        """
        # Retrieve matrix grid and its dictionary
        mdg: pp.MixedDimensionalGrid = self.model.mdg
        sd_matrix, d_matrix = mdg.subdomains(return_data=True)[0]

        # Retrieve indices
        cell_idx = self.get_region_indices(where="cc")

        # Manually obtain the exact pressure gradient
        x, y, z = sym.symbols("x y z")

        # Pressure gradient as a list of symbolic variables
        grad_p_matrix_sym = [
            [sym.diff(p, x), sym.diff(p, y), sym.diff(p, z)]
            for p in self.p_matrix
        ]

        # Lambdified list of pressure gradients
        grad_p_matrix_fun = [
            [
                sym.lambdify((x, y, z), gradp[0], "numpy"),
                sym.lambdify((x, y, z), gradp[1], "numpy"),
                sym.lambdify((x, y, z), gradp[2], "numpy"),
            ]
            for gradp in grad_p_matrix_sym
        ]

        # Obtain reconstructed pressure and create list of coefficients
        recon_p = d_matrix["estimates"]["recon_sd_pressure"].copy()
        pr = mdnme.utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = quadpy.t2.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_matrix)

        # Compute the true error for each subregion
        integral = np.zeros(sd_matrix.num_cells)
        for gradp, idx in zip(grad_p_matrix_fun, cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                # Exact pressure gradient in x, y and z
                gradp_exact_x = gradp[0](x[0], x[1], x[2])
                gradp_exact_y = gradp[1](x[0], x[1], x[2])
                gradp_exact_z = gradp[2](x[0], x[1], x[2])
                # Reconstructed pressure gradient in x, y and z
                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[0])
                gradp_recon_z = pr[2] * np.ones_like(x[0])
                # Integral in x, y and z
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                int_z = (gradp_exact_z - gradp_recon_z) ** 2
                return int_x + int_y + int_z

            integral += method.integrate(integrand, elements) * idx

        return integral

    def true_error_fracture(self) -> np.ndarray:
        """Compute true error contribution of the fracture subdomain.

        Returns:
            True error contribution of the fracture, containing the local error squared
            of each element of the grid. Shape is `sd.num_cells`.

        Note:
            - We assume the field ``d_matrix["estimates"]["recon_sd_pressure"]`` is
              available.
            - We employ a numerical quadrature scheme that is accurate up to a
              polynomial degree 10.

        """
        mdg = self.model.mdg
        sd_frac, d_frac = mdg.subdomains(return_data=True)[1]
        sd_rot = mdnme.RotatedGrid(sd_frac)

        # Declare symbolic variables
        y, z = sym.symbols("y z")

        # Compute symbolic and lambdified pressure gradient
        grad_p_frac_sym = [
            sym.diff(self.p_frac, y),
            sym.diff(self.p_frac, z)
        ]
        grad_p_frac_fun = [
            sym.lambdify((y, z), gradp, "numpy") for gradp in grad_p_frac_sym
        ]

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = d_frac["estimates"]["recon_sd_pressure"].copy()
        pr = mdnme.utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = quadpy.t2.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_frac, sd_rot)
        elements *= -1  # we have to use the physical coordinates here

        # Compute true error
        def integrand(x):
            # Exact pressure gradient
            gradp_exact_rot_y = -grad_p_frac_fun[0](x[0], x[1])  # -1 due to rotation
            gradp_exact_rot_z = -grad_p_frac_fun[1](x[0], x[1])  # -1 due to rotation
            # Reconstructed pressure gradients
            gradp_recon_y = pr[1] * np.ones_like(x[0])
            gradp_recon_z = pr[2] * np.ones_like(x[0])
            # Integral
            int_y = (gradp_exact_rot_y - gradp_recon_y) ** 2
            int_z = (gradp_exact_rot_z - gradp_recon_z) ** 2
            return int_y + int_z

        integral = method.integrate(integrand, elements)

        return integral

    def true_error_interface(self) -> np.ndarray:
        """Compute true error contribution of the interface.

        Returns:
            True error contribution of the interface, containing the local error squared
            of each element of the grid. Shape is `intf.num_cells`.

        """

        from mdnme.estimates.diffusive_error import (
            _get_high_pressure_trace,
            _get_low_pressure,
        )

        # Retrieve grids and dictionaries
        mdg: pp.MixedDimensionalGrid = self.model.mdg
        sd_high, data_high = mdg.subdomains(return_data=True)[0]
        sd_low, data_low = mdg.subdomains(return_data=True)[1]
        intf, data_intf = mdg.interfaces(return_data=True)[0]

        # Face-cell map between higher- and lower-dimensional subdomains
        frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
        frac_cells = sps.find(intf.secondary_to_mortar_avg())[1]

        # Obtain high and low-dimensional pressures
        recon_tracep_high = _get_high_pressure_trace(
            sd_low,
            sd_high,
            data_high,
            frac_faces,
        )
        recon_p_low = _get_low_pressure(data_low, frac_cells)
        recon_deltap = recon_p_low - recon_tracep_high

        # Retrieve side-grids tuples
        sides = intf.project_to_side_grids()

        values = []
        for side in sides:
            # Get projector and sidegrid object
            projector = side[0]
            sidegrid = side[1]

            # Declare integration method
            method = quadpy.t2.get_good_scheme(10)
            sidegrid_rot = mdnme.RotatedGrid(sidegrid)
            elements = mdnme.utils.get_quadpy_elements(sidegrid, sidegrid_rot)
            elements *= -1  # We need to use real coordinates

            # Project relevant quantities to the side grid
            recon_deltap_side = projector * recon_deltap

            # Retrieve exact 2D pressure
            y, z = sym.symbols("y z")
            exact_p_2d = sym.lambdify((y, z), self.p_frac, "numpy")
            exact_deltap_side = exact_p_2d  # high-dim pressure trace is zero

            # Declare integrand
            def integrand(x):
                recon_p_jump = mdnme.utils.evaluate_p1(
                    recon_deltap_side,
                    -x,  # negate due to rotation
                )
                return (exact_deltap_side(x) - recon_p_jump) ** 2

            # Compute integral
            diffusive_error_side = method.integrate(integrand, elements)

            # Append to list of values
            values.append(diffusive_error_side)

        return np.hstack(values)
