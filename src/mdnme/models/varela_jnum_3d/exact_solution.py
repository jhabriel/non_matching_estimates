"""
Module containing a slightly modified solution as given in Appendix D.2. from [1].

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

import mdnme


class VarelaJNumExactSolution3D:
    """Class containing the exact manufactured solution for the verification model."""

    def __init__(self, model):
        self.model = model

        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Smoothness coefficient
        n = 1.5

        # Distance and bubble functions
        distance_fun = [
            ((x - 0.5) ** 2 + (y - 0.25) ** 2 + (z - 0.25) ** 2) ** 0.5,  # bottom front
            ((x - 0.5) ** 2 + (y - 0.25) ** 2) ** 0.5,  # bottom middle
            ((x - 0.5) ** 2 + (y - 0.25) ** 2 + (z - 0.75) ** 2) ** 0.5,  # bottom back
            ((x - 0.5) ** 2 + (z - 0.25) ** 2) ** 0.5,  # front
            ((x - 0.5) ** 2) ** 0.5,  # middle
            ((x - 0.5) ** 2 + (z - 0.75) ** 2) ** 0.5,  # back
            ((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.25) ** 2) ** 0.5,  # top front
            ((x - 0.5) ** 2 + (y - 0.75) ** 2) ** 0.5,  # top middle
            ((x - 0.5) ** 2 + (y - 0.75) ** 2 + (z - 0.75) ** 2) ** 0.5,  # top back
        ]
        bubble_fun = (
            (y - 0.25) ** 2 * (y - 0.75) ** 2 * (z - 0.25) ** 2 * (z - 0.75) ** 2
        )

        # Exact pressure in the matrix
        p_matrix = [
            distance_fun[0] ** (1 + n),  # bottom front
            distance_fun[1] ** (1 + n),  # bottom middle
            distance_fun[2] ** (1 + n),  # bottom back
            distance_fun[3] ** (1 + n),  # front
            distance_fun[4] ** (1 + n) + bubble_fun * distance_fun[4],  # middle
            distance_fun[5] ** (1 + n),  # back
            distance_fun[6] ** (1 + n),  # top front
            distance_fun[7] ** (1 + n),  # top middle
            distance_fun[8] ** (1 + n),  # top back
        ]

        # Exact Darcy flux in the matrix
        q_matrix = [
            [-sym.diff(p, x), -sym.diff(p, y), -sym.diff(p, z)] for p in p_matrix
        ]

        # Exact divergence of the Darcy flux in the matrix
        f_matrix = [
            sym.diff(q[0], x) + sym.diff(q[1], y) + sym.diff(q[2], z) for q in q_matrix
        ]

        # Override f_matrix[4] (middle region) with the analytically simplified Laplacian.
        # The auto-diff of bubble_fun * |x-0.5| produces d²/dx²(|x-0.5|) written as
        # (u)^{-1/2} - u*(u)^{-3/2} (where u=(x-0.5)²), which is algebraically 0 but
        # evaluates to ∞ - ∞ = NaN in numpy at x ≈ 0.5. The true Laplacian of p_matrix[4]
        # = |x-0.5|^{1+n} + B(y,z)*|x-0.5| is:
        #   Δp_4 = (1+n)·n · ((x-0.5)²)^{(n-1)/2} + ((x-0.5)²)^{1/2} · (B_yy + B_zz)
        # Both terms are numerically stable everywhere, including at x = 0.5.
        _r2 = (x - sym.Rational(1, 2)) ** 2
        _by = (y - sym.Rational(1, 4)) ** 2 * (y - sym.Rational(3, 4)) ** 2
        _bz = (z - sym.Rational(1, 4)) ** 2 * (z - sym.Rational(3, 4)) ** 2
        f_matrix[4] = -(
            (1 + n) * n * _r2 ** sym.Rational(1, 4)
            + _r2 ** sym.Rational(1, 2)
            * (sym.diff(_by, y, 2) * _bz + _by * sym.diff(_bz, z, 2))
        )

        # Exact flux on the interface (mortar fluxes)
        q_intf = bubble_fun

        # Exact pressure in the fracture
        p_frac = -bubble_fun

        # Exact Darcy flux in the fracture
        q_frac = [-sym.diff(p_frac, y), -sym.diff(p_frac, z)]

        # Exact source in the fracture
        f_frac = sym.diff(q_frac[0], y) + sym.diff(q_frac[1], z) - 2 * q_intf

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

    def get_region_indices(self, where: str) -> list[np.ndarray]:
        """Get indices of the cells belonging to the different regions of the domain.

        Parameters:
            where: Use "cc" to evaluate at the cell centers, "fc" to evaluate at
                the face centers and "bg" to evaluate at the face centers associated
                with the external boundary

        Returns:
            List of length 9, containing the indices of the different regions of the
            domain.

        """
        # Sanity check
        assert where in ["cc", "fc", "bg"]

        # Retrieve coordinates
        sd = self.model.mdg.subdomains()[0]
        if where == "cc":
            x = sd.cell_centers
        elif where == "fc":
            x = sd.face_centers
        else:
            bg = self.model.mdg.subdomain_to_boundary_grid(sd)
            assert bg is not None
            x = bg.cell_centers

        # Get indices
        bottom_front = (x[1] < 0.25) & (x[2] < 0.25)
        bottom_middle = (x[1] < 0.25) & (x[2] > 0.25) & (x[2] < 0.75)
        bottom_back = (x[1] < 0.25) & (x[2] > 0.75)
        front = (x[1] > 0.25) & (x[1] < 0.75) & (x[2] < 0.25)
        middle = (x[1] >= 0.25) & (x[1] <= 0.75) & (x[2] >= 0.25) & (x[2] <= 0.75)
        back = (x[1] > 0.25) & (x[1] < 0.75) & (x[2] > 0.75)
        top_front = (x[1] > 0.75) & (x[2] < 0.25)
        top_middle = (x[1] > 0.75) & (x[2] > 0.25) & (x[2] < 0.75)
        top_back = (x[1] > 0.75) & (x[2] > 0.75)

        cell_idx = [
            bottom_front,
            bottom_middle,
            bottom_back,
            front,
            middle,
            back,
            top_front,
            top_middle,
            top_back,
        ]

        return cell_idx

    def matrix_pressure(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix pressure [Pa] at the cell centers.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, z), p, "numpy") for p in self.p_matrix]

        # Cell-centered pressures
        p_cc = np.zeros(sd_matrix.num_cells)
        for p, idx in zip(p_fun, cell_idx):
            p_cc += p(cc[0], cc[1], cc[2]) * idx

        return p_cc

    def matrix_flux(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Evaluate exact matrix Darcy flux [m^3 * s^-1] at the face centers.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_faces, )`` containing the exact Darcy
            fluxes at the face centers.

        Note:
            The returned fluxes are already scaled with ``sd_matrix.face_normals``.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of face indices
        fc = sd_matrix.face_centers
        face_idx = self.get_region_indices(where="fc")

        # Lambdify bubble function
        bubble_fun = sym.lambdify((y, z), self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((x, y, z), q[0], "numpy"),
                sym.lambdify((x, y, z), q[1], "numpy"),
                sym.lambdify((x, y, z), q[2], "numpy"),
            ]
            for q in self.q_matrix
        ]

        # Computation of the fluxes in the middle region results in NaN on faces that
        # are outside the middle region. We therefore need to first evaluate the
        # middle region and then the other regions so that NaN faces outside the middle
        # region can be overwritten accordingly.
        fn = sd_matrix.face_normals
        q_fc = np.zeros(sd_matrix.num_faces)

        q_fun_sorted = q_fun.copy()
        q_fun_sorted.pop(4)
        q_fun_sorted.insert(0, q_fun[4])

        face_idx_sorted = face_idx.copy()
        face_idx_sorted.pop(4)
        face_idx_sorted.insert(0, face_idx[4])

        # Perform evaluations using the sorted list of exact Darcy velocities
        for q, idx in zip(q_fun_sorted, face_idx_sorted):
            q_fc[idx] = (
                q[0](fc[0][idx], fc[1][idx], fc[2][idx]) * fn[0][idx]
                + q[1](fc[0][idx], fc[1][idx], fc[2][idx]) * fn[1][idx]
                + q[2](fc[0][idx], fc[1][idx], fc[2][idx]) * fn[2][idx]
            )

        # Now the only NaN faces should correspond to the internal boundaries, e.g.,
        # the ones at x = 0.5. To correct these values, we exploit the fact that
        # (rho_matrix * q_matrix) \dot n = (rho_intf * q_intf) holds in a continuous
        # sense. Furthermore, for our problem, rho_matrix = rho_intf = 1.0 at x = 0.5
        # and 0.25 <= y <= 0.75, 0.25 <= z <= 0.75 . Thus, the previous equality can
        # be simplified to q_matrix \dot n = q_intf on the internal boundaries.

        # Here, we cannot use the face normals since we'll get wrong signs (not
        # entirely sure why). Instead, we multiply by the face area and the face sign.
        frac_faces = np.where(sd_matrix.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[1][frac_faces], fc[2][frac_faces])
            * sd_matrix.face_areas[frac_faces]
            * sd_matrix.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    def fracture_pressure(self, sd_frac: pp.Grid) -> np.ndarray:
        """Evaluate exact fracture pressure at the cell centers.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variable
        y, z = sym.symbols("y z")

        # Cell centers
        cc = sd_frac.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify((y, z), self.p_frac, "numpy")

        # Evaluate at the cell centers
        p_cc = p_fun(cc[1], cc[2])

        return p_cc

    def fracture_flux(
        self,
        sd_frac: pp.Grid,
    ) -> np.ndarray:
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
        y, z = sym.symbols("y z")

        # Face centers and face normals
        fc = sd_frac.face_centers
        fn = sd_frac.face_normals

        # Lambdify expression
        q_fun = [sym.lambdify((y, z), q, "numpy") for q in self.q_frac]

        # Evaluate at the face centers and scale with face normals
        q_fc = q_fun[0](fc[1], fc[2]) * fn[1] + q_fun[1](fc[1], fc[2]) * fn[2]

        return q_fc

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
        y, z = sym.symbols("y z")

        # Cell centers and volumes
        cc = intf.cell_centers
        vol = intf.cell_volumes

        # Lambdify expression
        lmbda_fun = sym.lambdify((y, z), self.q_intf, "numpy")

        # Evaluate and "integrate"
        lmbda_cc = lmbda_fun(cc[1], cc[2]) * vol

        return lmbda_cc

    def boundary_values(self, boundary_grid_matrix: pp.BoundaryGrid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            boundary_grid_matrix: Matrix boundary grid.

        Returns:
            Array of ``shape=(bg.num_cells, )`` with the exact
            pressure values at the exterior boundary faces.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of face indices
        fc = boundary_grid_matrix.cell_centers
        face_idx = self.get_region_indices(where="bg")

        # Lambdify expression
        p_fun = [sym.lambdify((x, y, z), p, "numpy") for p in self.p_matrix]

        # Boundary pressures
        p_bf = np.zeros(boundary_grid_matrix.num_cells)
        for p, idx in zip(p_fun, face_idx):
            p_bf += p(fc[0], fc[1], fc[2]) * idx

        return p_bf

    def integrated_matrix_source(self, sd_matrix: pp.Grid) -> np.ndarray:
        """Obtain numerically integrated exact matrix sources.

        Parameters:
            sd_matrix: Matrix grid.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` with the exact integrated source
            terms for the matrix.

        Note:
            We employ a quadrature rule that is exact up to a polynomial degree 10.

        """
        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of indices
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, z), f, "numpy") for f in self.f_matrix]

        # Declare integration method and get hold of elements in QuadPy format
        int_method = quadpy.t3.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_matrix)

        integral = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            # Declare integrand
            def integrand(x: np.ndarray) -> np.ndarray:
                return f(x[0], x[1], x[2]) * np.ones_like(x[0])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def integrated_fracture_source(self, sd_frac: pp.Grid) -> np.ndarray:
        """Obtain numerically integrated exact fracture sources.

        Parameters:
            sd_frac: Fracture grid.

        Returns:
            Array of ``shape=(sd_frac.num_cells, )`` with the exact integrated source
            terms of the fracture.

        Note:
            We employ a quadrature rule that is exact upt to a polynomial of degree 10.

        """
        # Symbolic variables
        y, z = sym.symbols("y z")

        # Lambdify expression
        f_fun = sym.lambdify((y, z), self.f_frac, "numpy")

        # Obtain elements and declare integration method
        sd_rot = mdnme.RotatedGrid(sd_frac)
        method = quadpy.t2.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_frac, sd_rot)

        # Now we have to make sure we retrieve the correct physical coordinates
        # in reduced dimension.

        # We first check if we need to multiply the `elements` by -1 or not.
        if not np.all(np.sign(elements)):
            elements *= -1

        # Now we need retrieve the correct indices of the coordinate dimensions
        check = np.allclose(
            np.abs(sd_frac.cell_centers[1]),
            np.abs(sd_rot.cell_centers[0]),
            atol=1e-8,
            rtol=1e-5,
        )
        if check:
            ydim = 0  # y is the first reduced dimension
            zdim = 1  # z is the second reduced dimension
        else:
            ydim = 1  # y is the second reduced dimension
            zdim = 0  # z is the first reduced dimension

        def integrand(x):
            return f_fun(x[ydim], x[zdim])

        integral = method.integrate(integrand, elements)

        return integral

    def residual_error_matrix(self, sd_matrix: pp.Grid, d_matrix: dict) -> np.ndarray:
        """Compute square of residual errors for the (3D) host domain.

        Parameters:
            sd_matrix: Matrix grid.
            d_matrix: Dictionary containing the post-processed data. In particular,
                      we expect the field ``d_matrix["estimates"]["recon_sd_flux"]``
                      to be accessible.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the difference
            between the divergence of the reconstructed flux and the source term.

        Note:
            - We use a numerical integration scheme that is accurate for polynomials
              up to degree 10.

        """

        # Symbolic variables
        x, y, z = sym.symbols("x y z")

        # Get list of cell indices
        cell_idx = self.get_region_indices(where="cc")

        # Lambdify expression
        f_fun = [sym.lambdify((x, y, z), f, "numpy") for f in self.f_matrix]

        # Retrieve reconstructed subdomain flux and manually compute its divergence
        recon_u = d_matrix["estimates"]["recon_sd_flux"].copy()  # copy just in case
        u = mdnme.utils.poly2col(recon_u)
        div_u = 3 * u[0]

        # Integration method and retrieving elements
        int_method = quadpy.t3.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_matrix)

        # Local Poincare weights
        weights = (sd_matrix.cell_diameters() / np.pi) ** 2

        # Compute the integrals
        integral = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            # Declare integrand
            def integrand(x: np.ndarray) -> np.ndarray:
                return (f(x[0], x[1], x[2]) * np.ones_like(x[0]) - div_u) ** 2

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return weights * integral

    def residual_error_fracture(self, sd_frac: pp.Grid, d_frac: dict) -> np.ndarray:
        """Compute square of residual errors for the fracture (2D) grio.

        Parameters:
            sd_frac:  Fracture grid.
            d_frac:   Dictionary containing the post-processed data. In particular,
                      we expect the fields ``d_frac["estimates"]["recon_sd_flux"]``
                      and ``d_frac["estimates"]["sources_from_intf"]`` to be accessible.

        Returns:
            Array of ``shape=(sd_matrix.num_cells, )`` containing the difference
            between the divergence of the reconstructed flux and the source term.

        Note:
            - We use a numerical integration scheme that is accurate for polynomials
              up to degree 10.
        """
        # Integration method and retrieving elements
        y, z = sym.symbols("y z")

        # Retrieve reconstructed velocity and compute its divergence
        recon_u = d_frac["estimates"]["recon_sd_flux"].copy()
        u = mdnme.utils.poly2col(recon_u)  # coefficients of the reconstructed flux
        div_u = 2 * u[0]  # divergence of the reconstructed flux

        # Contribution from interface fluid fluxes to mass balance equation
        sources_from_intf = d_frac["estimates"]["sources_from_intf"].copy()

        # Lambdify expression
        f_fun = sym.lambdify((y, z), self.f_frac, "numpy")

        # Now we have to make sure we retrieve the correct physical coordinates
        # in reduced dimension.

        # Obtain elements and declare integration method
        sd_rot = mdnme.RotatedGrid(sd_frac)
        method = quadpy.t2.get_good_scheme(10)
        elements = mdnme.utils.get_quadpy_elements(sd_frac, sd_rot)

        # We first check if we need to multiply the `elements` by -1 or not.
        if not np.all(np.sign(elements)):
            elements *= -1

        # Now we need retrieve the correct indices of the coordinate dimensions
        check = np.allclose(
            np.abs(sd_frac.cell_centers[1]),
            np.abs(sd_rot.cell_centers[0]),
            atol=1e-8,
            rtol=1e-5,
        )
        if check:
            ydim = 0  # y is the first reduced dimension
            zdim = 1  # z is the second reduced dimension
        else:
            ydim = 1  # y is the second reduced dimension
            zdim = 0  # z is the first reduced dimension

        # Local Poincare weights
        weights = (sd_frac.cell_diameters() / np.pi) ** 2

        # Compute the integrals
        def integrand(x):
            return (f_fun(x[ydim], x[zdim]) - div_u + sources_from_intf) ** 2

        integral = method.integrate(integrand, elements)

        return weights * integral
