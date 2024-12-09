from typing import Literal

import numpy as np
import porepy as pp
import quadpy
import sympy as sym

import mdamr as amr


class TwoCrossingExactSolution:
    """Class containing the exact manufactured solution for the verification setup.

    Note:
    -----

          Numbering of subdomains           Numbering of interfaces
        ----------------------------       ----------------------------
        |                        6 |       |         12   5           |
        |           2|             |       |          | | |           |
        |            |             |       |   11 --- | *1| --- 6     |
        |      5 --- *1 --- 3      |       |      -- 4* * *2 --       |
        |            |             |       |   10 --- | *3| --- 7     |
        |          4 |             |       |          | | |           |
        |                          |       |         9    8           |
        ----------------------------       ----------------------------
    """

    def __init__(self):
        """Constructor of the class."""

        # Symbolic variables
        self.x, self.y = sym.symbols("x y")

        # Smoothness exponent
        n = 3
        epsilon = 1

        # --> Distance functions
        dist_vert_frac = [
            ((self.x - 0.5) ** 2 + (self.y - 0.75) ** 2) ** 0.5,  # top
            ((self.x - 0.5) ** 2 + (self.y - 0.75) ** 2) ** 0.5,  # top
            ((self.x - 0.5) ** 2 + (self.y - 0.75) ** 2) ** 0.5,  # top
            ((self.x - 0.5) ** 2) ** 0.5,  # middle
            ((self.x - 0.5) ** 2) ** 0.5,  # middle
            ((self.x - 0.5) ** 2) ** 0.5,  # middle
            ((self.x - 0.5) ** 2 + (self.y - 0.25) ** 2) ** 0.5,  # bottom
            ((self.x - 0.5) ** 2 + (self.y - 0.25) ** 2) ** 0.5,  # bottom
            ((self.x - 0.5) ** 2 + (self.y - 0.25) ** 2) ** 0.5,  # bottom
        ]
        dist_hor_frac = [
            ((self.x - 0.25) ** 2 + (self.y - 0.5) ** 2) ** 0.5,  # left
            ((self.y - 0.5) ** 2) ** 0.5,  # middle
            ((self.x - 0.75) ** 2 + (self.y - 0.5) ** 2) ** 0.5,  # right
            ((self.x - 0.25) ** 2 + (self.y - 0.5) ** 2) ** 0.5,  # left
            ((self.y - 0.5) ** 2) ** 0.5,  # middle
            ((self.x - 0.75) ** 2 + (self.y - 0.5) ** 2) ** 0.5,  # right
            ((self.x - 0.25) ** 2 + (self.y - 0.5) ** 2) ** 0.5,  # left
            ((self.y - 0.5) ** 2) ** 0.5,  # middle
            ((self.x - 0.75) ** 2 + (self.y - 0.5) ** 2) ** 0.5,  # right
        ]

        # --> Bubble functions
        bubble_vert_frac = [
            sym.Float(0),  # top
            sym.Float(0),  # top
            sym.Float(0),  # top
            (self.y - 0.25) ** 2 * (self.y - 0.75) ** 2,  # middle
            (self.y - 0.25) ** 2 * (self.y - 0.75) ** 2,  # middle
            (self.y - 0.25) ** 2 * (self.y - 0.75) ** 2,  # middle
            sym.Float(0),  # bottom
            sym.Float(0),  # bottom
            sym.Float(0),  # bottom
        ]
        bubble_hor_frac = [
            sym.Float(0),  # left
            (self.x - 0.25) ** 2 * (self.x - 0.75) ** 2,  # middle
            sym.Float(0),  # right
            sym.Float(0),  # left
            (self.x - 0.25) ** 2 * (self.x - 0.75) ** 2,  # middle
            sym.Float(0),  # right
            sym.Float(0),  # left
            (self.x - 0.25) ** 2 * (self.x - 0.75) ** 2,  # middle
            sym.Float(0),  # right
        ]

        # --> Exact pressure in the matrix
        partial_p_vert_frac = [
            d ** (1 + n) + epsilon * b * d
            for (d, b) in zip(dist_vert_frac, bubble_vert_frac)
        ]
        partial_p_hor_frac = [
            d ** (1 + n) + epsilon * b * d
            for (d, b) in zip(dist_hor_frac, bubble_hor_frac)
        ]
        p_6 = [
            pver * phor for (pver, phor) in zip(partial_p_vert_frac, partial_p_hor_frac)
        ]

        # --> Exact Darcy flux in the matrix
        q_6 = [[-sym.diff(p, self.x), -sym.diff(p, self.y)] for p in p_6]

        # --> Exact source in the matrix
        f_6 = [sym.diff(q[0], self.x) + sym.diff(q[1], self.y) for q in q_6]

        # --> Exact one-dimensional interface fluxes
        tol = 1e-10  # tolerance for the evaluation of exact interface fluxes
        # We multiply either by -1 or 1 depending on the sign of the normal vector
        lmbda_5 = sym.nsimplify(q_6[4][0].subs({"x": 0.5 + tol}), tolerance=1e-8) * -1
        lmbda_6 = sym.nsimplify(q_6[4][1].subs({"y": 0.5 + tol}), tolerance=1e-8) * -1
        lmbda_7 = sym.nsimplify(q_6[4][1].subs({"y": 0.5 - tol}), tolerance=1e-8) * 1
        lmbda_8 = sym.nsimplify(q_6[4][0].subs({"x": 0.5 + tol}), tolerance=1e-8) * -1
        lmbda_9 = sym.nsimplify(q_6[4][0].subs({"x": 0.5 - tol}), tolerance=1e-8) * 1
        lmbda_10 = sym.nsimplify(q_6[4][1].subs({"y": 0.5 - tol}), tolerance=1e-8) * 1
        lmbda_11 = sym.nsimplify(q_6[4][1].subs({"y": 0.5 + tol}), tolerance=1e-8) * -1
        lmbda_12 = sym.nsimplify(q_6[4][0].subs({"x": 0.5 - tol}), tolerance=1e-8) * 1

        assert lmbda_5 == lmbda_12
        assert lmbda_6 == lmbda_7
        assert lmbda_8 == lmbda_9
        assert lmbda_10 == lmbda_11

        # ---> Exact high-dimensional pressure traces on internal boundaries
        tr_p_6_at_5 = sym.nsimplify(p_6[4].subs({"x": 0.5}), tolerance=1e-8)
        tr_p_6_at_6 = sym.nsimplify(p_6[4].subs({"y": 0.5}), tolerance=1e-8)
        tr_p_6_at_7 = sym.nsimplify(p_6[4].subs({"y": 0.5}), tolerance=1e-8)
        tr_p_6_at_8 = sym.nsimplify(p_6[4].subs({"x": 0.5}), tolerance=1e-8)
        tr_p_6_at_9 = sym.nsimplify(p_6[4].subs({"x": 0.5}), tolerance=1e-8)
        tr_p_6_at_10 = sym.nsimplify(p_6[4].subs({"y": 0.5}), tolerance=1e-8)
        tr_p_6_at_11 = sym.nsimplify(p_6[4].subs({"y": 0.5}), tolerance=1e-8)
        tr_p_6_at_12 = sym.nsimplify(p_6[4].subs({"x": 0.5}), tolerance=1e-8)

        assert tr_p_6_at_5 == tr_p_6_at_12
        assert tr_p_6_at_6 == tr_p_6_at_7
        assert tr_p_6_at_8 == tr_p_6_at_9
        assert tr_p_6_at_10 == tr_p_6_at_11

        # --> Exact one-dimensional pressures
        p_2 = -lmbda_5 + tr_p_6_at_5
        p_3 = -lmbda_6 + tr_p_6_at_6
        p_4 = -lmbda_8 + tr_p_6_at_8
        p_5 = -lmbda_10 + tr_p_6_at_10

        # --> Exact one-dimensional Darcy fluxes
        q_2 = -sym.diff(p_2, self.y)
        q_3 = -sym.diff(p_3, self.x)
        q_4 = -sym.diff(p_4, self.y)
        q_5 = -sym.diff(p_5, self.x)

        # Check whether we have zero boundary conditions on the fracture tips
        np.testing.assert_almost_equal(q_2.subs({"y": 0.75}), 0)
        np.testing.assert_almost_equal(q_3.subs({"x": 0.75}), 0)
        np.testing.assert_almost_equal(q_4.subs({"y": 0.25}), 0)
        np.testing.assert_almost_equal(q_5.subs({"x": 0.25}), 0)

        # --> Exact one-dimensional sources
        f_2 = sym.diff(q_2, self.y) - (lmbda_5 + lmbda_12)
        f_3 = sym.diff(q_3, self.x) - (lmbda_6 + lmbda_7)
        f_4 = sym.diff(q_4, self.y) - (lmbda_8 + lmbda_9)
        f_5 = sym.diff(q_5, self.x) - (lmbda_10 + lmbda_11)

        # --> Exact zero-dimensional interface fluxes
        lmbda_1 = -q_2.subs({"y": 0.50 - tol})
        lmbda_2 = -q_3.subs({"x": 0.50 - tol})
        lmbda_3 = q_4.subs({"y": 0.50 + tol})
        lmbda_4 = q_5.subs({"x": 0.50 + tol})

        np.testing.assert_almost_equal(lmbda_1, lmbda_2)
        np.testing.assert_almost_equal(lmbda_1, lmbda_3)
        np.testing.assert_almost_equal(lmbda_1, lmbda_4)
        np.testing.assert_almost_equal(lmbda_2, lmbda_3)
        np.testing.assert_almost_equal(lmbda_2, lmbda_4)
        np.testing.assert_almost_equal(lmbda_3, lmbda_4)

        # Traces of pressures at (0.5, 0.5)
        tr_p2 = p_2.subs({"y": 0.50})
        tr_p3 = p_3.subs({"x": 0.50})
        tr_p4 = p_4.subs({"y": 0.50})
        tr_p5 = p_5.subs({"x": 0.50})

        np.testing.assert_almost_equal(tr_p2, tr_p3)
        np.testing.assert_almost_equal(tr_p2, tr_p4)
        np.testing.assert_almost_equal(tr_p2, tr_p5)
        np.testing.assert_almost_equal(tr_p3, tr_p4)
        np.testing.assert_almost_equal(tr_p3, tr_p5)
        np.testing.assert_almost_equal(tr_p4, tr_p5)

        # Zero-dimensional pressure
        # Due to the symmetry of the problem, all the following
        # expressions result in a unique zero-dimensional pressure
        p1_from_2 = tr_p2 - lmbda_1
        p1_from_3 = tr_p3 - lmbda_2
        p1_from_4 = tr_p4 - lmbda_3
        p1_from_5 = tr_p5 - lmbda_4

        np.testing.assert_almost_equal(p1_from_2, p1_from_3)
        np.testing.assert_almost_equal(p1_from_2, p1_from_4)
        np.testing.assert_almost_equal(p1_from_2, p1_from_5)
        np.testing.assert_almost_equal(p1_from_3, p1_from_4)
        np.testing.assert_almost_equal(p1_from_3, p1_from_5)
        np.testing.assert_almost_equal(p1_from_4, p1_from_5)

        p_1 = np.mean(np.array([p1_from_2, p1_from_3, p1_from_4, p1_from_5]))

        # Exact zero-dimensional source term
        f_1 = -1 * (lmbda_1 + lmbda_2 + lmbda_3 + lmbda_4)

        # Public attributes

        # Subdomain, 2d
        self.p_6 = p_6
        self.q_6 = q_6
        self.f_6 = f_6

        # Subdomains, 1d
        self.p_2 = p_2
        self.p_3 = p_3
        self.p_4 = p_4
        self.p_5 = p_5
        self.q_2 = q_2
        self.q_3 = q_3
        self.q_4 = q_4
        self.q_5 = q_5
        self.f_2 = f_2
        self.f_3 = f_3
        self.f_4 = f_4
        self.f_5 = f_5

        # Subdomain, 0d
        self.p_1 = p_1
        self.f_1 = f_1

        # Interfaces, 1d
        self.lmbda_5 = lmbda_5
        self.lmbda_6 = lmbda_6
        self.lmbda_7 = lmbda_7
        self.lmbda_8 = lmbda_8
        self.lmbda_9 = lmbda_9
        self.lmbda_10 = lmbda_10
        self.lmbda_11 = lmbda_11
        self.lmbda_12 = lmbda_12

        # Interfaces, 0d
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.lmbda_3 = lmbda_3
        self.lmbda_4 = lmbda_4

        # # Bubble function
        # self._bubble = bubble_fun
        # self._distance = distance_fun

    # ----> Debugging
    def _distance_function(self, sd_2d: pp.Grid) -> np.ndarray:
        d_fun = [sym.lambdify((self.x, self.y), d, "numpy") for d in self._distance]
        d_cc = np.zeros(sd_2d.num_cells)
        for d, idx in zip(d_fun, self.get_2d_subregions(sd_2d, "cc")):
            d_cc += d(sd_2d.cell_centers[0], sd_2d.cell_centers[1]) * idx
        return d_cc

    def _bubble_function(self, sd_2d: pp.Grid) -> np.ndarray:
        b_fun = [sym.lambdify((self.x, self.y), b, "numpy") for b in self._bubble]
        b_cc = np.zeros(sd_2d.num_cells)
        for b, idx in zip(b_fun, self.get_2d_subregions(sd_2d, "cc")):
            b_cc += b(sd_2d.cell_centers[0], sd_2d.cell_centers[1]) * idx
        return b_cc

    def _matrix_flux(self, sd_2d: pp.Grid, which="magnitude") -> np.ndarray:
        # Retrieve indices for the different subregions
        cell_idx = self.get_2d_subregions(sd_2d, "cc")
        cc = sd_2d.cell_centers

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((self.x, self.y), q[0], "numpy"),
                sym.lambdify((self.x, self.y), q[1], "numpy"),
            ]
            for q in self.q_6
        ]

        q_cc_horizontal = np.zeros(sd_2d.num_cells)
        q_cc_vertical = np.zeros(sd_2d.num_cells)

        # Perform evaluations using the sorted list of exact Darcy velocities
        for q, idx in zip(q_fun, cell_idx):
            q_cc_horizontal[idx] = q[0](cc[0][idx], cc[1][idx])
            q_cc_vertical[idx] = q[1](cc[0][idx], cc[1][idx])

        q_magnitude = np.sqrt(q_cc_horizontal**2 + q_cc_vertical**2)

        if which == "horizontal":
            q_out = q_cc_horizontal
        elif which == "vertical":
            q_out = q_cc_vertical
        elif which == "magnitude":
            q_out = q_magnitude
        else:
            raise NotImplementedError()

        return q_out

    # ----> Subregions
    def get_2d_subregions(
        self, g: pp.Grid | pp.BoundaryGrid, which: Literal["cc", "fc"]
    ):
        """
        Return boolean indices of the different subregions of the matrix.

        :param g: Matrix grid or matrix boundary grid.
        :param which: Whether to retrieve subregions for faces or cells.
        :raises ValueError: If a boundary grid is given but a face center value is asked.
        :return: List of boolean arrays denote whether the face or center belong
                 to a subregion.
        """
        # Sanity check
        if isinstance(g, pp.BoundaryGrid) and which == "fc":
            raise ValueError(
                "Face center values for boundary grid does not make sense."
            )

        if which == "cc":
            xc = g.cell_centers
        elif which == "fc":
            xc = g.face_centers
        else:
            raise NotImplementedError()

        subregion_1 = (xc[0] < 0.25) & (xc[1] > 0.75)
        subregion_2 = (xc[0] > 0.25) & (xc[0] < 0.75) & (xc[1] > 0.75)
        subregion_3 = (xc[0] > 0.75) & (xc[1] > 0.75)
        subregion_4 = (xc[0] < 0.25) & (xc[1] > 0.25) & (xc[1] < 0.75)
        subregion_5 = (xc[0] > 0.25) & (xc[0] < 0.75) & (xc[1] > 0.25) & (xc[1] < 0.75)
        subregion_6 = (xc[0] > 0.75) & (xc[1] > 0.25) & (xc[1] < 0.75)
        subregion_7 = (xc[0] < 0.25) & (xc[1] < 0.25)
        subregion_8 = (xc[0] > 0.25) & (xc[0] < 0.75) & (xc[1] < 0.25)
        subregion_9 = (xc[0] > 0.75) & (xc[1] < 0.25)

        subregion_indices = [
            subregion_1,
            subregion_2,
            subregion_3,
            subregion_4,
            subregion_5,
            subregion_6,
            subregion_7,
            subregion_8,
            subregion_9,
        ]

        return subregion_indices

    def plot_subregions(self, sd_matrix: pp.Grid, subregion_id: int = None) -> None:
        subregions = np.zeros(sd_matrix.num_cells)
        cc_idx_list = self.get_2d_subregions(sd_matrix, which="cc")
        for idx, cc_idx in enumerate(cc_idx_list):
            subregions[cc_idx] = idx + 1

        if subregion_id is None:
            pp.plot_grid(sd_matrix, subregions, plot_2d=True)
        else:
            subregion = np.zeros_like(subregions)
            subregion[cc_idx_list[subregion_id - 1]] = 1
            pp.plot_grid(sd_matrix, subregion, plot_2d=True)

    # ----> Pressure
    def p_0d(self, sd_0d: pp.Grid) -> np.ndarray:
        return np.array([self.p_1], dtype=np.float32)

    def p_1d_north(self, sd_1d_north: pp.Grid) -> np.ndarray:
        assert sd_1d_north.label == "north"
        p_fun = sym.lambdify(self.y, self.p_2, "numpy")
        return p_fun(sd_1d_north.cell_centers[1])

    def p_1d_west(self, sd_1d_west: pp.Grid) -> np.ndarray:
        assert sd_1d_west.label == "west"
        p_fun = sym.lambdify(self.x, self.p_3, "numpy")
        return p_fun(sd_1d_west.cell_centers[0])

    def p_1d_south(self, sd_1d_south: pp.Grid) -> np.ndarray:
        assert sd_1d_south.label == "south"
        p_fun = sym.lambdify(self.y, self.p_4, "numpy")
        return p_fun(sd_1d_south.cell_centers[1])

    def p_1d_east(self, sd_1d_east: pp.Grid) -> np.ndarray:
        assert sd_1d_east.label == "east"
        p_fun = sym.lambdify(self.x, self.p_5, "numpy")
        return p_fun(sd_1d_east.cell_centers[0])

    def p_2d(self, sd_2d: pp.Grid) -> np.ndarray:
        p_fun = [sym.lambdify((self.x, self.y), p, "numpy") for p in self.p_6]
        p_cc = np.zeros(sd_2d.num_cells)
        for p, idx in zip(p_fun, self.get_2d_subregions(sd_2d, "cc")):
            p_cc += p(sd_2d.cell_centers[0], sd_2d.cell_centers[1]) * idx
        return p_cc

    # ----> Darcy fluxes
    def q_1d_north(self, sd_1d_north: pp.Grid) -> np.ndarray:
        q_fun = sym.lambdify(self.y, self.q_2, "numpy")
        return q_fun(sd_1d_north.face_centers[1])

    def q_1d_west(self, sd_1d_west: pp.Grid) -> np.ndarray:
        q_fun = sym.lambdify(self.x, self.q_3, "numpy")
        return q_fun(sd_1d_west.face_centers[0])

    def q_1d_south(self, sd_1d_south: pp.Grid) -> np.ndarray:
        q_fun = sym.lambdify(self.y, self.q_4, "numpy")
        return q_fun(sd_1d_south.face_centers[1])

    def q_1d_east(self, sd_1d_east: pp.Grid) -> np.ndarray:
        q_fun = sym.lambdify(self.x, self.q_5, "numpy")
        return q_fun(sd_1d_east.face_centers[1])

    def q_2d(self, sd_2d: pp.Grid) -> np.ndarray:
        # TODO: This probably must be fixed

        # Retrieve indices for the different subregions
        face_idx = self.get_2d_subregions(sd_2d, "fc")

        # Lambdify bubble
        bubble_fun = sym.lambdify((self.x, self.y), self._bubble, "numpy")

        # Lambdify expression
        q_fun = [
            [
                sym.lambdify((self.x, self.y), q[0], "numpy"),
                sym.lambdify((self.x, self.y), q[1], "numpy"),
            ]
            for q in self.q_6
        ]

        # Computation of the fluxes in the middle region results in NaN on faces that
        # are outside the middle region. We therefore need to first evaluate the middle
        # region and then the other regions, so that NaN faces outside the middle
        # region can be overwritten accordingly.
        fc = sd_2d.face_centers
        fn = sd_2d.face_normals
        q_fc = np.zeros(sd_2d.num_faces)

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
        frac_faces = np.where(sd_2d.tags["fracture_faces"])[0]
        q_fc[frac_faces] = (
            bubble_fun(fc[0][frac_faces], fc[1][frac_faces])
            * sd_2d.face_areas[frac_faces]
            * sd_2d.signs_and_cells_of_boundary_faces(frac_faces)[0]
        )

        return q_fc

    # ----> Interface fluxes
    def lmbda_0d_north(self, intf_0d_north: pp.MortarGrid) -> np.ndarray:
        return np.array([self.lmbda_1], dtype=np.float32)

    def lmbda_0d_west(self, intf_0d_west: pp.MortarGrid) -> np.ndarray:
        return np.array([self.lmbda_2], dtype=np.float32)

    def lmbda_0d_south(self, intf_0d_south: pp.MortarGrid) -> np.ndarray:
        return np.array([self.lmbda_3], dtype=np.float32)

    def lmbda_0d_east(self, intf_0d_east: pp.MortarGrid) -> np.ndarray:
        return np.array([self.lmbda_4], dtype=np.float32)

    def lmbda_1d_north(self, intf_1d_north: pp.MortarGrid) -> np.ndarray:
        lmbda_fun = sym.lambdify(self.y, self.lmbda_5, "numpy")  # same as lmbda_12
        vol = intf_1d_north.cell_volumes
        return lmbda_fun(intf_1d_north.cell_centers[1]) * vol

    def lmbda_1d_west(self, intf_1d_west: pp.MortarGrid) -> np.ndarray:
        lmbda_fun = sym.lambdify(self.x, self.lmbda_7, "numpy")  # same as lmbda_6
        vol = intf_1d_west.cell_volumes
        return lmbda_fun(intf_1d_west.cell_centers[0]) * vol

    def lmbda_1d_south(self, intf_1d_south: pp.MortarGrid) -> np.ndarray:
        lmbda_fun = sym.lambdify(self.y, self.lmbda_9, "numpy")  # same as lmbda_8
        vol = intf_1d_south.cell_volumes
        return lmbda_fun(intf_1d_south.cell_centers[1]) * vol

    def lmbda_1d_east(self, intf_1d_east: pp.MortarGrid) -> np.ndarray:
        lmbda_fun = sym.lambdify(self.x, self.lmbda_10, "numpy")  # same as lmbda_11
        vol = intf_1d_east.cell_volumes
        return lmbda_fun(intf_1d_east.cell_centers[0]) * vol

    # ----> Integrated source terms
    def f_0d(self, sd_0d: pp.Grid) -> np.ndarray:
        return np.array([self.f_1], dtype=np.float32)

    def f_1d_north(self, sd_1d_north: pp.Grid) -> np.ndarray:
        f_fun = sym.lambdify(self.y, self.f_2, "numpy")

        method = quadpy.c1.newton_cotes_open(10)
        elements = amr.utils.get_quadpy_elements(sd_1d_north)
        elements *= 1  # we have to use the real `y` coordinates here, no sign changes

        def integrand(y):
            return f_fun(y)

        integral = method.integrate(integrand, elements)
        assert sd_1d_north.label == "north"
        # # Approximate integral
        # cc = sd_1d_north.cell_centers
        # vol = sd_1d_north.cell_volumes
        # integral = f_fun(cc[1]) * vol

        return integral

    def f_1d_west(self, sd_1d_west: pp.Grid) -> np.ndarray:
        f_fun = sym.lambdify(self.x, self.f_5, "numpy")

        method = quadpy.c1.newton_cotes_open(10)
        elements = amr.utils.get_quadpy_elements(sd_1d_west)
        elements *= 1  # we have to use the real `x` coordinates here, no sign change

        def integrand(x):
            return f_fun(x)

        integral = method.integrate(integrand, elements)

        # Approximate integral
        assert sd_1d_west.label == "west"
        # cc = sd_1d_west.cell_centers
        # vol = sd_1d_west.cell_volumes
        # integral = f_fun(cc[0]) * vol

        return integral

    def f_1d_south(self, sd_1d_south: pp.Grid) -> np.ndarray:
        f_fun = sym.lambdify(self.y, self.f_4, "numpy")

        method = quadpy.c1.newton_cotes_open(10)
        elements = amr.utils.get_quadpy_elements(sd_1d_south)
        elements *= -1  # we have to use the real `y` coordinates here, change sign

        def integrand(y):
            return f_fun(y)

        integral = method.integrate(integrand, elements)

        # Approximate integral
        assert sd_1d_south.label == "south"
        # cc = sd_1d_south.cell_centers
        # vol = sd_1d_south.cell_volumes
        # integral = f_fun(cc[1]) * vol

        return integral

    def f_1d_east(self, sd_1d_east: pp.Grid) -> np.ndarray:
        f_fun = sym.lambdify(self.x, self.f_3, "numpy")

        method = quadpy.c1.newton_cotes_open(10)
        elements = amr.utils.get_quadpy_elements(sd_1d_east)
        elements *= 1  # we have to use the real `x` coordinates here, no sign changes

        def integrand(x):
            return f_fun(x)

        integral = method.integrate(integrand, elements)

        # Approximate integral
        assert sd_1d_east.label == "east"
        # cc = sd_1d_east.cell_centers
        # vol = sd_1d_east.cell_volumes
        # integral = f_fun(cc[0]) * vol

        return integral

    def f_2d(self, sd_2d: pp.Grid) -> np.ndarray:
        f_fun = [sym.lambdify((self.x, self.y), f, "numpy") for f in self.f_6]

        # Declare integration method and get hold of elements in QuadPy format
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd_2d)

        integral = np.zeros(sd_2d.num_cells)
        for f, idx in zip(f_fun, self.get_2d_subregions(sd_2d, "cc")):
            # Declare integrand
            def integrand(x):
                return f(x[0], x[1]) * np.ones_like(x[0])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        # # Symbolic variables
        # x, y = sym.symbols("x y")
        #
        # # Get list of cell indices
        # cc = sd_2d.cell_centers
        # cell_idx = self.get_2d_subregions(sd_2d, "cc")
        #
        # # Lambdify expression
        # f_fun = [sym.lambdify((self.x, self.y), f, "numpy") for f in self.f_6]
        #
        # # Integrated cell-centered sources
        # vol = sd_2d.cell_volumes
        # f_cc = np.zeros(sd_2d.num_cells)
        # for f, idx in zip(f_fun, cell_idx):
        #     f_cc += f(cc[0], cc[1]) * vol * idx

        return integral

    def boundary_values(self, bg_2d: pp.BoundaryGrid) -> np.ndarray:
        p_fun = [sym.lambdify((self.x, self.y), p, "numpy") for p in self.p_6]

        # Boundary pressures
        p_bf = np.zeros(bg_2d.num_cells)
        for p, idx in zip(p_fun, self.get_2d_subregions(bg_2d, "cc")):
            p_bf += (
                p(
                    bg_2d.cell_centers[0],
                    bg_2d.cell_centers[1],
                )
                * idx
            )

        return p_bf

    def residual_error_0d(self, sd_0d: pp.Grid) -> np.ndarray:
        ...

    def residual_error_1d_north(self, sd_1d_north: pp.Grid) -> np.ndarray:
        ...

    def residual_error_1d_west(self, sd_1d_west: pp.Grid) -> np.ndarray:
        ...

    def residual_error_1d_south(self, sd_1d_south: pp.Grid) -> np.ndarray:
        ...

    def residual_error_1d_east(self, sd_1d_east: pp.Grid) -> np.ndarray:
        ...

    def residual_error_2d(self, sd_2d: pp.Grid) -> np.ndarray:
        ...

    # def residual_error_matrix(self, sd_matrix: pp.Grid, d_matrix: dict) -> np.ndarray:
    #     """Compute square of residual errors for 2D (only the norm)"""
    #     # Symbolic variables
    #     x, y = sym.symbols("x y")
    #
    #     # Get list of cell indices
    #     cc = sd_matrix.cell_centers
    #     bot = cc[1] < 0.25
    #     mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
    #     top = cc[1] > 0.75
    #     cell_idx = [bot, mid, top]
    #
    #     # Lambdify expression
    #     f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_matrix]
    #
    #     # Retrieve reconstructed velocity and compute divergence
    #     recon_u = d_matrix["estimates"]["recon_sd_flux"].copy()
    #     u = amr.utils.poly2col(recon_u)
    #     div_u = 2 * u[0]
    #
    #     # Integration method and retrieving elements
    #     int_method = quadpy.t2.get_good_scheme(10)
    #     elements = amr.utils.get_quadpy_elements(sd_matrix)
    #
    #     integral = np.zeros(sd_matrix.num_cells)
    #     weights = (sd_matrix.cell_diameters() / np.pi) ** 2
    #     for f, idx in zip(f_fun, cell_idx):
    #         # Declare integrand
    #         def integrand(x):
    #             return (f(x[0], x[1]) * np.ones_like(x[0]) - div_u) ** 2
    #
    #         # Integrate, and add the contribution of each subregion
    #         integral += int_method.integrate(integrand, elements) * idx
    #
    #     return weights * integral
    #
    # def residual_error_fracture(self, sd_frac: pp.Grid, d_frac: dict) -> np.ndarray:
    #     """Compute square of residual errors for 2D (only the norm)"""
    #
    #     # Retrieve reconstructed velocity and compute its divergence
    #     recon_u = d_frac["estimates"]["recon_sd_flux"].copy()
    #     u = amr.utils.poly2col(recon_u)
    #     div_u = u[0]
    #
    #     # Contribution from interface fluid fluxes to mass balance equation
    #     sources_from_intf = d_frac["estimates"]["sources_from_intf"].copy()
    #
    #     # Integration method and retrieving elements
    #     y = sym.symbols("y")
    #
    #     # Lambdify expression
    #     f_fun = sym.lambdify(y, self.f_frac, "numpy")
    #
    #     method = quadpy.c1.newton_cotes_closed(10)
    #     elements = amr.utils.get_quadpy_elements(sd_frac)
    #     elements *= -1  # we have to use the real `y` coordinates here
    #
    #     weights = (sd_frac.cell_diameters() / np.pi) ** 2
    #
    #     def integrand(y):
    #         return (f_fun(y) - div_u + sources_from_intf) ** 2
    #
    #     integral = method.integrate(integrand, elements)
    #
    #     return weights * integral
