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
    Class containing the computation of the true errors of the first numerical example.
    """

    def __init__(self):
        super().__init__()

    def true_error(self, mdg: pp.MixedDimensionalGrid) -> float:
        """
        Compute global true error.

        Parameters:
            mdg: pp.MixedDimensionalGrid
                Mixed-dimensional grid of the problem.

        Returns:
            Value of the true error for the whole mixed-dimensional grid.

        """
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

    def true_error_matrix(self, sd_matrix: pp.Grid, d_matrix: dict) -> np.ndarray:
        """
        Compute true error contribution of the matrix subdomain.

        Parameters:
            sd_matrix: pp.Grid
                Matrix subdomain grid.
            d_matrix: dict
                Data dictionary of the matrix. We assume that the numerical pressure
                has been reconstructed, i.e., that d["estimates"][
                "recon_sd_pressure"] is available.

        Returns:
            True error contribution of the matrix, containing the local error squared
            of each element of the grid. Shape is `sd.num_cells`.

        """
        # ---> Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # ---> Obtain exact pressure gradient
        x, y = sym.symbols("x y")

        grad_p_matrix_sym = [[sym.diff(p, x), sym.diff(p, y)] for p in self.p_matrix]

        grad_p_matrix_fun = [
            [
                sym.lambdify((x, y), gradp[0], "numpy"),
                sym.lambdify((x, y), gradp[1], "numpy"),
            ]
            for gradp in grad_p_matrix_sym
        ]

        # ---> Obtain reconstructed pressure and create list of coefficients
        recon_p = d_matrix["estimates"]["recon_sd_pressure"]
        pr = mdnme.utils.poly2col(recon_p)

        # ---> Obtain elements and declare integration method
        method = quadpy.t2.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_matrix)

        # ---> Compute the true error for each subregion
        integral = np.zeros(sd_matrix.num_cells)
        for gradp, idx in zip(grad_p_matrix_fun, cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x):
                # Exact pressure gradient in x and y
                gradp_exact_x = gradp[0](x[0], x[1])
                gradp_exact_y = gradp[1](x[0], x[1])
                # Reconstructed pressure gradient in x and y
                gradp_recon_x = pr[0] * np.ones_like(x[0])
                gradp_recon_y = pr[1] * np.ones_like(x[1])
                # Integral in x and y
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2
                return int_x + int_y

            integral += method.integrate(integrand, elements) * idx

        return integral

    def true_error_fracture(self, sd_fracture: pp.Grid, d_fracture: dict) -> np.ndarray:
        """
        Compute true error contribution of the fracture subdomain.

        Parameters:
            sd_fracture: pp.Grid
                Fracture subdomain grid.
            d_fracture: dict
                Data dictionary of the fracture. We assume that the numerical pressure
                has been reconstructed, i.e., that d["estimates"][
                "recon_sd_pressure"] is available.

        Returns:
            True error contribution of the fracture, containing the local error squared
            of each element of the grid. Shape is `sd.num_cells`.

        """
        # ---> Obtain exact pressure gradient
        y = sym.symbols("y")
        grad_p_frac_sym = sym.diff(self.p_frac, y)
        grad_p_frac_fun = sym.lambdify(y, grad_p_frac_sym, "numpy")

        # ---> Get hold of reconstructed pressure and create list of coefficients
        recon_p = d_fracture["estimates"]["recon_sd_pressure"]
        pr = mdnme.utils.poly2col(recon_p)

        # ---> Obtain elements and declare integration method
        method = quadpy.c1.newton_cotes_closed(10)
        elements = mdnme.utils.get_quadpy_elements(sd_fracture)
        elements *= -1  # we have to use the real `y` coordinates here

        # ---> Compute true error
        def integrand(x):
            # Exact pressure gradient
            gradp_exact_rot = -grad_p_frac_fun(x)  # -1 due to rotation
            # Reconstructed pressure gradient
            gradp_recon_x = pr[0] * np.ones_like(x[0])
            # Intregral
            int_x = (gradp_exact_rot - gradp_recon_x) ** 2
            return int_x

        integral = method.integrate(integrand, elements)

        return integral

    def true_error_interface(
        self,
        intf: pp.MortarGrid,
        data_intf: dict,
        sd_high: pp.Grid,
        data_high: dict,
        sd_low: pp.Grid,
        data_low: dict,
    ) -> np.ndarray:
        """
        Compute true error contribution of the interface.

        Parameters:
            intf: pp.MortarGrid
                Interface (mortar) grid.
            data_intf: dict
                Data dictionary associated to the interface grid. Not really used
                here, but included for consistency.
            sd_high: pp.Grid
                Matrix subdomain grid.
            data_high: dict
                Data dictionary of the Matrix. We assume that the numerical pressure
                has been reconstructed, i.e., that d["estimates"][
                "recon_sd_pressure"] is available.
            sd_low: pp.Grid
                Fracture subdomain grid.
            data_low: dict
                Data dictionary of the fracture. We assume that the numerical pressure
                has been reconstructed, i.e., that d["estimates"][
                "recon_sd_pressure"] is available.

        Returns:
            True error contribution of the interface, containing the local error squared
            of each element of the grid. Shape is `intf.num_cells`.

        """

        from mdnme.estimates.diffusive_error import (
            _get_high_pressure_trace,
            _get_low_pressure,
        )

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
            method = quadpy.c1.newton_cotes_closed(10)
            elements = mdnme.utils.get_quadpy_elements(sidegrid)
            elements *= -1  # We need to use real coordinates

            # Project relevant quantities to the side grid
            recon_deltap_side = projector * recon_deltap

            # Retrieve exact 1d pressure
            y = sym.symbols("y")
            exact_p_1d = sym.lambdify(y, self.p_frac, "numpy")
            exact_deltap_side = exact_p_1d  # high-dim pressure trace is zero

            # Declare integrand
            def integrand(x):
                coors = x[np.newaxis, :, :]  # this is needed for 1D grids
                recon_p_jump = mdnme.utils.evaluate_p1(
                    recon_deltap_side,
                    -coors,  # negate due to rotation
                )
                return (exact_deltap_side(x) - recon_p_jump) ** 2

            # Compute integral
            diffusive_error_side = method.integrate(integrand, elements)

            # Append to list of values
            values.append(diffusive_error_side)

        return np.hstack(values)
