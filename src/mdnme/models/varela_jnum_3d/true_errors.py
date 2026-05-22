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
from mdnme.estimates.diffusive_error import _get_high_pressure_trace, _get_low_pressure
from mdnme.models.varela_jnum_3d.exact_solution import VarelaJNumExactSolution3D
from mdnme.utils.internal_boundary_grid import InternalBoundaryGrid
from mdnme.utils.primal_projections import (
    prolong_to_transfer,
    scott_zhang_quasi_interpolant,
)
from mdnme.utils.transfer_grid import TransferGrid


class VarelaJNumTrueErrors3D(VarelaJNumExactSolution3D):
    """
    Class containing the computation of true errors for the 3D manufactured solution.
    """

    def __init__(self, model):
        super().__init__(model)
        self.is_nonmatch = model.params.get("non_matching", False)

    def true_error_primal(self) -> float:
        """Compute global true error (mixed-dimensional majorant) for the primal var.

        Returns:
            Value of the true error for the whole mixed-dimensional grid.

        """
        te_sq_matrix = self.true_error_matrix_primal()
        te_sq_fracture = self.true_error_fracture_primal()
        te_sq_intf = self.true_error_interface_primal()

        return np.sqrt(te_sq_matrix.sum() + te_sq_fracture.sum() + te_sq_intf.sum())

    def true_error_matrix_primal(self) -> np.ndarray:
        """
        Compute true error contribution of the matrix subdomain for the primal var.

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
        sd_matrix, d_matrix = mdg.subdomains(return_data=True, dim=3)[0]

        # Retrieve indices
        cell_idx = self.get_region_indices(where="cc")

        # Manually obtain the exact pressure gradient
        x, y, z = sym.symbols("x y z")

        # Pressure gradient as a list of symbolic variables
        grad_p_matrix_sym = [
            [sym.diff(p, x), sym.diff(p, y), sym.diff(p, z)] for p in self.p_matrix
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

        # Override x-component of gradient for the middle region (index 4).
        # Auto-diff of bubble * |x-0.5| gives bubble*(x-0.5)/((x-0.5)^2)^{1/2},
        # which evaluates to 0/0 = NaN in numpy at x = 0.5. The stable form uses
        # np.sign, which returns 0 at x = 0.5 (correct limiting behavior).
        _n = 1.5

        def _grad4_x_stable(
            xv: np.ndarray, yv: np.ndarray, zv: np.ndarray
        ) -> np.ndarray:
            b = (yv - 0.25) ** 2 * (yv - 0.75) ** 2 * (zv - 0.25) ** 2 * (zv - 0.75) ** 2
            return (1 + _n) * (xv - 0.5) * ((xv - 0.5) ** 2) ** 0.25 + b * np.sign(xv - 0.5)

        grad_p_matrix_fun[4][0] = _grad4_x_stable

        # Obtain reconstructed pressure and create list of coefficients
        recon_p = d_matrix["estimates"]["recon_sd_pressure"]
        pr = mdnme.utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = quadpy.t3.get_good_scheme(12)
        elements = mdnme.utils.get_quadpy_elements(sd_matrix)

        # Compute the true error for each subregion
        integral = np.zeros(sd_matrix.num_cells)
        for gradp, idx in zip(grad_p_matrix_fun, cell_idx):
            # Declare integrand and add subregion contribution
            def integrand(x: np.ndarray) -> np.ndarray:
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

    def true_error_fracture_primal(self) -> np.ndarray:
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
        sd_frac, d_frac = mdg.subdomains(return_data=True, dim=2)[0]
        sd_rot = mdnme.RotatedGrid(sd_frac)

        # Declare symbolic variables
        y, z = sym.symbols("y z")

        # Compute symbolic and lambdified pressure gradient
        grad_p_frac_sym = [sym.diff(self.p_frac, y), sym.diff(self.p_frac, z)]
        grad_p_frac_fun = [
            sym.lambdify((y, z), gradp, "numpy") for gradp in grad_p_frac_sym
        ]

        # Get hold of reconstructed pressure and create list of coefficients
        recon_p = d_frac["estimates"]["recon_sd_pressure"]
        pr = mdnme.utils.poly2col(recon_p)

        # Obtain elements and declare integration method
        method = quadpy.t2.get_good_scheme(16)
        elements = mdnme.utils.get_quadpy_elements(sd_frac, sd_rot)
        num_cells = sd_frac.num_cells

        # Mapping from rotated to physical coordinates
        active = np.where(sd_rot.dim_bool)[0]
        inactive = np.where(~sd_rot.dim_bool)[0][0]
        P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        R = sd_rot.rotation_matrix
        T_yz = P_yz @ R.T[:, active]
        rot_cc_full = R @ sd_frac.cell_centers
        c_vec = rot_cc_full[inactive, :]
        n_yz = (P_yz @ R.T[:, inactive]).reshape(2, 1)
        b_yz = n_yz @ c_vec.reshape(1, num_cells)

        # Now we need retrieve the correct indices of the coordinate dimensions
        # Compute true error
        def integrand(x: np.ndarray) -> np.ndarray:
            # Reconstructed gradients
            gradp_recon_y = pr[0] * np.ones_like(x[0])
            gradp_recon_z = pr[1] * np.ones_like(x[0])

            # Exact pressure gradient
            yz = np.einsum("ab,bnm->anm", T_yz, x) + b_yz[:, :, None]
            gradp_exact_y = grad_p_frac_fun[0](yz[0], yz[1])
            gradp_exact_z = grad_p_frac_fun[1](yz[0], yz[1])
            # Now project to rotated basis
            Ty = T_yz.T  # (2,2)
            gradp_exact_rot_y = Ty[0, 0] * gradp_exact_y + Ty[0, 1] * gradp_exact_z
            gradp_exact_rot_z = Ty[1, 0] * gradp_exact_y + Ty[1, 1] * gradp_exact_z
            # Integral
            int_y = (gradp_exact_rot_y - gradp_recon_y) ** 2
            int_z = (gradp_exact_rot_z - gradp_recon_z) ** 2
            return int_y + int_z

        integral = method.integrate(integrand, elements)

        return integral

    def true_error_interface_primal(self) -> np.ndarray:
        """Compute true error contribution of the interface."""
        if not self.is_nonmatch:
            return self._true_error_interface_matching_primal()
        else:
            return self._true_error_interface_nonmatching_primal()

    def _true_error_interface_nonmatching_primal(self) -> np.ndarray:
        """Compute true error contribution of the interface for nonmatching grids"""
        tol = 1e-8  # geometric tolerance

        # Retrieve grids and dictionaries
        mdg: pp.MixedDimensionalGrid = self.model.mdg
        sd_high, data_high = mdg.subdomains(return_data=True)[0]
        sd_low, data_low = mdg.subdomains(return_data=True)[1]
        intf, data_intf = mdg.interfaces(return_data=True)[0]

        # --- low-dim pressure (per-cell P1 on its own grid) ---
        p_low_frac = data_low["estimates"]["recon_sd_pressure"]  # (n_frac_cells, 3)

        # --- face-trace of high-dim pressure, in interface frame ---
        # NOTE: p_trace_high[i] corresponds to sd_high face index frac_faces[i]
        frac_faces = sps.find(intf.primary_to_mortar_avg())[1]
        p_trace_high = _get_high_pressure_trace(
            sd_low, sd_high, data_high, frac_faces
        )  # (n_frac_faces, 3)
        # map: high face id -> local index into frac_faces
        face2pos = {int(f): i for i, f in enumerate(frac_faces)}

        # --- IBG (per-side internal-boundary grids in interface frame) ---
        ibg = InternalBoundaryGrid(intf, sd_high, tol=tol)

        # --- quadrature on 2D mortar sides ---
        method = quadpy.t2.get_good_scheme(20)

        values = []
        # loop sides in the mortar’s canonical order
        for P_msg, mg_side in intf.project_to_side_grids():
            # identify the side enum that owns this mortar side grid
            side_enum = next(k for k, v in intf.side_grids.items() if v is mg_side)

            # IBG side grid and its parent faces (one parent face per IBG triangle)
            ibg_side = ibg.ibg_side_grid(side_enum)
            parent_faces = ibg.parent_face_of_cell(side_enum)  # shape: (n_ibg_cells,)

            # (1) IBG-side tr(p_high): assign face P1 coeffs to each IBG cell via
            # parent map
            if ibg_side.num_cells == 0:
                tr_hi_on_ibg = np.zeros((0, 3))
            else:
                idx = np.fromiter(
                    (face2pos[int(f)] for f in parent_faces),
                    dtype=int,
                    count=parent_faces.size,
                )
                tr_hi_on_ibg = p_trace_high[idx, :]  # (n_ibg_cells, 3)

            # (2) Transfer IBG→mortar-side and frac→mortar-side
            # (same canonical frame; no R needed)
            tg_ibg_msg = TransferGrid(g_source=ibg_side, g_target=mg_side, tol=tol)
            tg_fg_msg = TransferGrid(g_source=sd_low, g_target=mg_side, tol=tol)

            # Internal boundary side grid to mortar side grid pressure projection
            tracep_on_tg = prolong_to_transfer(tg_ibg_msg, tr_hi_on_ibg)
            tracep_on_msg = scott_zhang_quasi_interpolant(tg_ibg_msg, tracep_on_tg)

            # Fracture grid to mortar side grid pressure projection
            fracp_on_tg = prolong_to_transfer(tg_fg_msg, p_low_frac)
            fracp_on_msg = scott_zhang_quasi_interpolant(tg_fg_msg, fracp_on_tg)

            # 3b) Map rotated -> physical (y,z): y,z = T_yz @ xi + b_yz
            sidegrid_rot = mdnme.RotatedGrid(mg_side)
            R = sidegrid_rot.rotation_matrix  # x_rot = R @ x_phys
            active = np.where(sidegrid_rot.dim_bool)[0]  # two in-plane indices
            inactive = np.where(~sidegrid_rot.dim_bool)[0][0]  # one normal index
            P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            T_yz = P_yz @ R.T[:, active]  # (2,2)

            sidegrid = mg_side
            rot_cc_full = R @ sidegrid.cell_centers  # (3, N)
            c_vec = rot_cc_full[inactive, :]  # (N,)
            n_yz = (P_yz @ R.T[:, inactive]).reshape(2, 1)  # (2,1)
            b_yz = n_yz @ c_vec.reshape(1, sidegrid.num_cells)  # (2, N)

            # (4) jump and integration on mortar side
            deltap_side = fracp_on_msg - tracep_on_msg  # (n_msg_cells, 3)
            elements = mdnme.utils.get_quadpy_elements(mg_side)

            # (5) Retrieve exact 2D pressure
            y, z = sym.symbols("y z")
            exact_p_2d = sym.lambdify((y, z), self.p_frac, "numpy")
            exact_deltap_side = exact_p_2d  # high-dim pressure trace is zero

            # Declare integrand
            def integrand(x: np.ndarray) -> np.ndarray:
                # Evaluate reconstructed pressure jump at quadrature points
                c = mdnme.utils.poly2col(deltap_side)
                recon_p_jump = c[0] * x[0] + c[1] * x[1] + c[2]
                # Evaluate exact pressure jump at quadrature points
                yz = np.einsum("ab,bnm->anm", T_yz, x) + b_yz[:, :, None]  # (2, N, M)
                exact_jump = exact_deltap_side(yz[0], yz[1])  # (N, M)
                return (exact_jump - recon_p_jump) ** 2

            # Compute integral
            diffusive_error_side = method.integrate(integrand, elements)

            # Append to list of values
            values.append(diffusive_error_side)

        return np.hstack(values)

    def _true_error_interface_matching_primal(self) -> np.ndarray:
        """Compute true error contribution of the interface for matching grids.

        Returns:
            True error contribution of the interface, containing the local error squared
            of each element of the grid. Shape is `intf.num_cells`.

        """
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
            sidegrid_rot = mdnme.RotatedGrid(sidegrid)

            # Obtain elements and declare integration method
            method = quadpy.t2.get_good_scheme(20)
            elements = mdnme.utils.get_quadpy_elements(sidegrid, sidegrid_rot)

            # 3b) Map rotated -> physical (y,z): y,z = T_yz @ xi + b_yz
            R = sidegrid_rot.rotation_matrix  # x_rot = R @ x_phys
            active = np.where(sidegrid_rot.dim_bool)[0]  # two in-plane indices
            inactive = np.where(~sidegrid_rot.dim_bool)[0][0]  # one normal index
            P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            T_yz = P_yz @ R.T[:, active]  # (2,2)

            rot_cc_full = R @ sidegrid.cell_centers  # (3, N)
            c_vec = rot_cc_full[inactive, :]  # (N,)
            n_yz = (P_yz @ R.T[:, inactive]).reshape(2, 1)  # (2,1)
            b_yz = n_yz @ c_vec.reshape(1, sidegrid.num_cells)  # (2, N)

            # Project relevant quantities to the side grid
            recon_deltap_side = projector * recon_deltap

            # Retrieve exact 2D pressure
            y, z = sym.symbols("y z")
            exact_p_2d = sym.lambdify((y, z), self.p_frac, "numpy")
            exact_deltap_side = exact_p_2d  # high-dim pressure trace is zero

            # Declare integrand
            def integrand(x: np.ndarray) -> np.ndarray:
                # Evaluate reconstructed pressure jump at quadrature points
                c = mdnme.utils.poly2col(recon_deltap_side)
                recon_p_jump = c[0] * x[0] + c[1] * x[1] + c[2]
                # Evaluate exact pressure jump at quadrature points
                yz = np.einsum("ab,bnm->anm", T_yz, x) + b_yz[:, :, None]  # (2, N, M)
                exact_jump = exact_deltap_side(yz[0], yz[1])  # (N, M)
                return (exact_jump - recon_p_jump) ** 2

            # Compute integral
            diffusive_error_side = method.integrate(integrand, elements)

            # Append to list of values
            values.append(diffusive_error_side)

        return np.hstack(values)

    # --- Add these methods inside VarelaJNumTrueErrors3D ---

    # ===== Global aggregator (dual variable) ===================================
    def true_error_dual(self) -> float:
        """Global L2(true error) of the dual variable u on matrix, fracture, interface."""
        te_u_sq_mat = self.true_error_dual_matrix().sum()
        te_u_sq_frac = self.true_error_dual_fracture().sum()
        te_u_sq_intf = self.true_error_dual_interface().sum()
        return np.sqrt(te_u_sq_mat + te_u_sq_frac + te_u_sq_intf)

    # ===== Matrix (3D) =========================================================
    def true_error_dual_matrix(self) -> np.ndarray:
        """Per-cell ∫_K |u_exact - u_h|^2 in the 3D matrix."""
        mdg: pp.MixedDimensionalGrid = self.model.mdg
        sd_matrix, d_matrix = mdg.subdomains(return_data=True, dim=3)[0]

        # Lambdify exact q = [q_x, q_y, q_z] per region
        x, y, z = sym.symbols("x y z")
        q_fun = [
            [
                sym.lambdify((x, y, z), q[0], "numpy"),
                sym.lambdify((x, y, z), q[1], "numpy"),
                sym.lambdify((x, y, z), q[2], "numpy"),
            ]
            for q in self.q_matrix
        ]

        # Reconstructed u_h (assume constant-per-cell coefficients via poly2col)
        recon_u = d_matrix["estimates"]["recon_sd_flux"]
        u = mdnme.utils.poly2col(recon_u)

        # Quadrature / elements and region masks
        method = quadpy.t3.get_good_scheme(12)
        elements = mdnme.utils.get_quadpy_elements(sd_matrix)
        cell_idx = self.get_region_indices(where="cc")

        out = np.zeros(sd_matrix.num_cells)
        for qf, idx in zip(q_fun, cell_idx):

            def integrand(X: np.ndarray) -> np.ndarray:
                ux_e = qf[0](X[0], X[1], X[2])
                uy_e = qf[1](X[0], X[1], X[2])
                uz_e = qf[2](X[0], X[1], X[2])
                # broadcast u_h constants element-wise
                ux = u[0] * X[0] + u[1]
                uy = u[0] * X[1] + u[2]
                uz = u[0] * X[2] + u[3]
                return (ux_e - ux) ** 2 + (uy_e - uy) ** 2 + (uz_e - uz) ** 2

            out += method.integrate(integrand, elements) * idx

        return out

    # ===== Fracture (2D) =======================================================
    def true_error_dual_fracture(self) -> np.ndarray:
        """Per-cell ∫_T |u_exact - u_h|^2 on the 2D fracture grid (tangential yz)."""
        mdg: pp.MixedDimensionalGrid = self.model.mdg
        sd_frac, d_frac = mdg.subdomains(return_data=True, dim=2)[0]
        sd_rot = mdnme.RotatedGrid(sd_frac)

        # exact q_frac = [qy(y,z), qz(y,z)] in physical yz
        y, z = sym.symbols("y z")
        qy_fun = sym.lambdify((y, z), self.q_frac[0], "numpy")
        qz_fun = sym.lambdify((y, z), self.q_frac[1], "numpy")

        # reconstructed u_h in local (rotated) yz frame; use constant coeffs
        recon_u = d_frac["estimates"]["recon_sd_flux"]
        u = mdnme.utils.poly2col(recon_u)

        # Quad + elements (with rotation mapping you already use)
        method = quadpy.t2.get_good_scheme(16)
        elements = mdnme.utils.get_quadpy_elements(sd_frac, sd_rot)

        active = np.where(sd_rot.dim_bool)[0]
        inactive = np.where(~sd_rot.dim_bool)[0][0]
        P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        R = sd_rot.rotation_matrix
        T_yz = P_yz @ R.T[:, active]  # map local (ξ,η) -> physical (y,z)
        rot_cc_full = R @ sd_frac.cell_centers
        c_vec = rot_cc_full[inactive, :]
        n_yz = (P_yz @ R.T[:, inactive]).reshape(2, 1)
        b_yz = n_yz @ c_vec.reshape(1, sd_frac.num_cells)
        Ty = T_yz.T  # use same transform as your pressure routine

        def integrand(xi: np.ndarray) -> np.ndarray:
            # u_h constants
            uy = u[0] * xi[0] + u[1]
            uz = u[0] * xi[1] + u[2]
            # exact in physical yz
            yz = np.einsum("ab,bnm->anm", T_yz, xi) + b_yz[:, :, None]
            qy = qy_fun(yz[0], yz[1])
            qz = qz_fun(yz[0], yz[1])
            # rotate physical components to local (rotated) yz frame
            uy_e = Ty[0, 0] * qy + Ty[0, 1] * qz
            uz_e = Ty[1, 0] * qy + Ty[1, 1] * qz
            return (uy_e - uy) ** 2 + (uz_e - uz) ** 2

        return method.integrate(integrand, elements)

    # --- Interface (2D mortar): P0 integrated λ_h
    def true_error_dual_interface(self) -> np.ndarray:
        """Per-cell ∫_E (λ_exact - λ_h_density)^2 dS on the mortar grid.

        Notes:
            - Expects data_intf["estimates"]["fv_intf_flux"] to store *integrated*
              P0 mortar flux per cell (shape (Nc,) or (Nc,1)).
            - We densitize via λ_h_density = Λ_h / |E|.
            - Quadrature uses degree=20 since (bubble)^2 has degree 16.
        """
        mdg: pp.MixedDimensionalGrid = self.model.mdg
        intf, data_intf = mdg.interfaces(return_data=True)[0]

        # Reconstructed integrated flux Λ_h per mortar cell (P0)
        lam_h_int = data_intf["estimates"]["fv_intf_flux"]
        lam_h_int = np.asarray(lam_h_int).reshape(-1)

        # Densitize to get constant density per cell
        vol = intf.cell_volumes
        lam_h_dens_all = lam_h_int / vol  # shape (Nc,)

        # Exact density λ(y,z)
        y, z = sym.symbols("y z")
        lam_fun = sym.lambdify((y, z), self.q_intf, "numpy")

        values = []
        # Degree 20 to integrate (degree-16) squared bubble accurately
        method = quadpy.t2.get_good_scheme(20)

        for P_msg, sidegrid in intf.project_to_side_grids():
            # Map (ξ,η) on the side grid -> physical (y,z)
            side_rot = mdnme.RotatedGrid(sidegrid)
            R = side_rot.rotation_matrix
            active = np.where(side_rot.dim_bool)[0]
            inactive = np.where(~side_rot.dim_bool)[0][0]

            P_yz = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            T_yz = P_yz @ R.T[:, active]  # (2,2)

            rot_cc_full = R @ sidegrid.cell_centers
            c_vec = rot_cc_full[inactive, :]
            n_yz = (P_yz @ R.T[:, inactive]).reshape(2, 1)
            b_yz = n_yz @ c_vec.reshape(1, sidegrid.num_cells)  # (2, Nsides)

            # Elements and per-side constant λ_h density
            elements = mdnme.utils.get_quadpy_elements(sidegrid, side_rot)
            lam_h_dens_side = P_msg * lam_h_dens_all  # shape (Nsides,)

            def integrand(xi: np.ndarray) -> np.ndarray:
                # exact λ at quadrature points
                yz = np.einsum("ab,bnm->anm", T_yz, xi) + b_yz[:, :, None]
                lam_e = lam_fun(yz[0], yz[1])  # (Nsides, Nq)
                # reconstructed constant density per side cell, broadcast over points
                lam_h = lam_h_dens_side[:, None] * np.ones_like(xi[0])
                return (lam_e - lam_h) ** 2

            values.append(method.integrate(integrand, elements))

        return np.hstack(values)
