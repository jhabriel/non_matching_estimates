"""
Module containing the exact solution as given in Appendix D.1. from [1].

Reference:
    - [1] Varela, J., Ahmed, E., Keilegavlen, E., Nordbotten, J. & Radu, F. (2023). A
      posteriori error estimates for hierarchical mixed-dimensional elliptic
      equations. Journal of Numerical Mathematics, 31(4), 247-280.
      https://doi.org/10.1515/jnma-2022-0038

"""

import numpy as np
import porepy as pp
import quadpy
import sympy as sym

import mdnme as amr


class Varela2023JNumExactSolution2d:
    """Class containing the exact manufactured solution for the verification setup."""

    def __init__(self):
        """Constructor of the class."""

        # Symbolic variables
        x, y = sym.symbols("x y")

        # Smoothness exponent
        n = 1.5

        # Distance and bubble functions
        distance_fun = [
            ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5,  # bottom
            ((x - 0.5) ** 2) ** 0.5,  # middle
            ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5,  # top
        ]
        bubble_fun = (y - 0.25) ** 2 * (y - 0.75) ** 2

        # Exact pressure in the matrix
        p_matrix = [
            distance_fun[0] ** (1 + n),
            distance_fun[1] ** (1 + n) + bubble_fun * distance_fun[1],
            distance_fun[2] ** (1 + n),
        ]

        # Exact Darcy flux in the matrix
        q_matrix = [[-sym.diff(p, x), -sym.diff(p, y)] for p in p_matrix]

        # Exact source in the matrix
        f_matrix = [sym.diff(q[0], x) + sym.diff(q[1], y) for q in q_matrix]

        # Exact interface flux
        q_intf = bubble_fun

        # Exact pressure in the fracture
        p_frac = -bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = -sym.diff(p_frac, y)

        # Exact source in the fracture
        f_frac = sym.diff(q_frac, y) - 2 * q_intf

        # Public attributes
        self.p_matrix = p_matrix
        self.q_matrix = q_matrix
        self.f_matrix = f_matrix
        self.p_frac = p_frac
        self.q_frac = q_frac
        self.f_frac = f_frac
        self.q_intf = q_intf

        # Private attributes
        self._bubble = bubble_fun

    def matrix_pressure(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix pressure at the cell centers.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y), p, "numpy") for p in self.p_matrix]

        # Cell-centered pressures
        p_cc = np.zeros(sd_matrix.num_cells)
        for p, idx in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1]) * idx

        return p_cc

    def matrix_flux(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix Darcy flux at the face centers.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_matrix.face_normals``.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = sd_matrix.face_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify bubble
        bubble_fun = sym.lambdify(y, self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [sym.lambdify((x, y), q[0], "numpy"), sym.lambdify((x, y), q[1], "numpy")]
            for q in self.q_matrix
        ]

        # Computation of the fluxes in the middle region results in NaN on faces that
        # are outside the middle region. We therefore need to first evaluate the middle
        # region and then the other regions, so that NaN faces outside the middle
        # region can be overwritten accordingly.
        fn = sd_matrix.face_normals
        q_fc = np.zeros(sd_matrix.num_faces)

        q_fun_sorted = q_fun.copy()
        q_fun_sorted.pop(1)
        q_fun_sorted.insert(0, q_fun[1])

        face_idx_sorted = face_idx.copy()
        face_idx_sorted.pop(1)
        face_idx_sorted.insert(0, face_idx[1])

        # Perform evaluations using the sorted list of exact Darcy velocities
        for q, idx in zip(q_fun_sorted, face_idx_sorted):
            q_fc[idx] = (
                q[0](fc[0][idx], fc[1][idx]) * fn[0][idx]
                + q[1](fc[0][idx], fc[1][idx]) * fn[1][idx]
            )

        # We need to correct the values of the exact Darcy fluxes at the internal
        # boundaries since they evaluate to NaN due to a division by zero.
        # What we do is to exploit the fact that trace(q) = \lambda holds in a
        # continuous sense, and use that expression on internal boundaries instead.
        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_matrix.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces])
            * sd_matrix.face_areas[frac_faces]
            * sd_matrix.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def matrix_source(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Compute exact integrated matrix source.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_matrix]

        # Integrated cell-centered sources
        vol = sd_matrix.cell_volumes
        f_cc = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            f_cc += f(cc[0], cc[1]) * vol * idx

        return f_cc

    def fracture_pressure(self, sd_frac: pp.Grid) -> np.ndarray:
        """Evaluate exact fracture pressure at the cell centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify(y, self.p_frac, "numpy")

        # Evaluate at the cell centers
        p_cc = p_fun(cc[1])

        return p_cc

    def fracture_flux(self, sd_frac: pp.Grid) -> np.ndarray:
        """Evaluate exact fracture Darcy flux at the face centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_face.face_normals``.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Face centers and face normals
        fc = sd_frac.face_centers
        fn = sd_frac.face_normals

        # Lambdify expression
        q_fun = sym.lambdify(y, self.q_frac, "numpy")

        # Evaluate at the face centers and scale with face normals
        q_fc = q_fun(fc[1]) * fn[1]

        return q_fc

    def fracture_source(self, sd_frac: pp.Grid) -> np.ndarray:
        """Compute exact integrated fracture source.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact integrated
            sources.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Cell centers and volumes
        cc = sd_frac.cell_centers
        vol = sd_frac.cell_volumes

        # Lambdify expression
        f_fun = sym.lambdify(y, self.f_frac, "numpy")

        # Evaluate and integrate
        f_cc = f_fun(cc[1]) * vol

        return f_cc

    def interface_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Compute exact mortar fluxes at the interface.

        Parameters:
            intf: Mortar grid.

        Returns:
            Array of ``shape=(intf.num_cells, )`` containing the exact mortar fluxes.

        Note:
            The returned mortar fluxes are already scaled with ``intf.cell_volumes``.

        """
        # Symbolic variable
        y = sym.symbols("y")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes

        # Lambdify expression
        lmbda_fun = sym.lambdify(y, self.q_intf, "numpy")

        # Evaluate and integrate
        lmbda_cc = lmbda_fun(cc[1]) * vol

        return lmbda_cc

    def boundary_values(self, boundary_grid_matrix: pp.BoundaryGrid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            boundary_grid_matrix: Matrix boundary grid.

        Returns:
            Array of ``shape=(boundary_grid_matrix.num_cells, )`` with the exact
            pressure values at the exterior boundary faces.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = boundary_grid_matrix.cell_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y), p, "numpy") for p in self.p_matrix]

        # Boundary pressures
        p_bf = np.zeros(boundary_grid_matrix.num_cells)
        for p, idx in zip(p_fun, face_idx):
            p_bf += p(fc[0], fc[1]) * idx

        return p_bf

    def integrated_matrix_source(self, sd_matrix: pp.Grid):
        """

        :param sd_matrix:
        :return:
        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_matrix]

        # Declare integration method and get hold of elements in QuadPy format
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd_matrix)

        integral = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            # Declare integrand
            def integrand(x):
                return f(x[0], x[1]) * np.ones_like(x[0])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def integrated_fracture_source(self, sd_frac):
        """

        :param sd_frac:
        :return:
        """
        # Symbolic variable
        y = sym.symbols("y")

        # Lambdify expression
        f_fun = sym.lambdify(y, self.f_frac, "numpy")

        method = quadpy.c1.newton_cotes_closed(10)
        elements = amr.utils.get_quadpy_elements(sd_frac)
        elements *= -1  # we have to use the real `y` coordinates here

        def integrand(y):
            return f_fun(y)

        integral = method.integrate(integrand, elements)

        return integral

    def residual_error_matrix(self, sd_matrix: pp.Grid, d_matrix: dict) -> np.ndarray:
        """Compute square of residual errors for 2D (only the norm)"""
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_matrix]

        # Retrieve reconstructed velocity and compute divergence
        recon_u = d_matrix["estimates"]["recon_sd_flux"].copy()
        u = amr.utils.poly2col(recon_u)
        div_u = 2 * u[0]

        # Integration method and retrieving elements
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd_matrix)

        integral = np.zeros(sd_matrix.num_cells)
        weights = (sd_matrix.cell_diameters() / np.pi) ** 2
        for f, idx in zip(f_fun, cell_idx):
            # Declare integrand
            def integrand(x):
                return (f(x[0], x[1]) * np.ones_like(x[0]) - div_u) ** 2

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return weights * integral

    def residual_error_fracture(self, sd_frac: pp.Grid, d_frac: dict) -> np.ndarray:
        """Compute square of residual errors for 2D (only the norm)"""

        # Retrieve reconstructed velocity and compute its divergence
        recon_u = d_frac["estimates"]["recon_sd_flux"].copy()
        u = amr.utils.poly2col(recon_u)
        div_u = u[0]

        # Contribution from interface fluid fluxes to mass balance equation
        sources_from_intf = d_frac["estimates"]["sources_from_intf"].copy()

        # Integration method and retrieving elements
        y = sym.symbols("y")

        # Lambdify expression
        f_fun = sym.lambdify(y, self.f_frac, "numpy")

        method = quadpy.c1.newton_cotes_closed(10)
        elements = amr.utils.get_quadpy_elements(sd_frac)
        elements *= -1  # we have to use the real `y` coordinates here

        weights = (sd_frac.cell_diameters() / np.pi) ** 2

        def integrand(y):
            return (f_fun(y) - div_u + sources_from_intf) ** 2

        integral = method.integrate(integrand, elements)

        return weights * integral
