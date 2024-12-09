from __future__ import annotations

import numpy as np


# %% Interpolation and polynomial-related functions
def interpolate_p1(point_val: np.ndarray, point_coo: np.ndarray):
    """
    Performs a linear local interpolation of a P1 FEM element given
    the pressure values and the coordinates at the Lagrangian nodes.

    Parameters
    ----------
    point_val : NumPy nd-array of shape (g.num_cells x num_Lagr_nodes)
        Pressures values at the Lagrangian nodes.
    point_coo : NumPy nd-array of shape (g.dim x g.num_cells x num_Lagr_nodes)
        Coordinates of the Lagrangian nodes. In the case of embedded entities,
        the points should correspond to the rotated coordinates.

    Raises
    ------
    Value Error
        If the number of columns of point_val is different from 4 (3D), 3 (2d),
        or 2 (1D)

    Returns
    -------
    coeff : Numpy nd-array of shape (g.num_cells x (g.dim+1))
        Coefficients of the cell-wise P1 polynomial satisfying:
        c0 x + c1                   (1D),
        c0 x + c1 y + c2            (2D),
        c0 x + c1 y + c3 z + c4     (3D).

    """

    # Get rows, cols, and dimensionality
    rows = point_val.shape[0]  # number of cells
    cols = point_val.shape[1]  # number of Lagrangian nodes per cell
    if cols == 4:
        dim = 3
    elif cols == 3:
        dim = 2
    elif cols == 2:
        dim = 1
    else:
        raise ValueError("P1 reconstruction only valid for 1d, 2d, and 3d.")

    if dim == 3:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        z = point_coo[2].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, y, z, ones])
        lcl = np.reshape(lcl, newshape=[rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, newshape=[rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    elif dim == 2:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, y, ones])
        lcl = np.reshape(lcl, newshape=[rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, newshape=[rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    else:
        x = point_coo.flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, ones])
        lcl = np.reshape(lcl, newshape=[rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, newshape=[rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    return coeff


def evaluate_p1(p1_coefficients: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Evaluates a P1 polynomial at the given coordinates

    Parameters
    ----------
        p1_coefficients (np.ndarray) : Polynomial to be evaluated, i.e., the one obtained from
            interpolate_P1. The expected shape is: rows x num_lagrangian_nodes.
        coordinates (np.ndarray): Coordinates with shape axes x rows x cols, where axes is
            the number of dimensions. If there is only one dimension present, we expect to
            to have 1 x row x cols nd-array.

    Raises
    ------
        ValueError: if there is any inconsistency in the shape of the inputs.

    Returns
    -------
        val (np.ndarray): Values of the polynomial at the coordinate points.
    """

    # Check if p1 coefficients has the correct shape
    if p1_coefficients.shape[1] not in [2, 3, 4]:
        raise ValueError("Number of coefficients does not match a P1 polynomial")

    # Check if coordinates has the correct number of dimensions
    if len(coordinates.shape) != 3:
        raise ValueError("Coordinates array must be three-dimensional")

    # Retrieve coefficients
    c = poly2col(p1_coefficients)

    if len(c) == 4:
        val = (
            c[0] * coordinates[0] + c[1] * coordinates[1] + c[2] * coordinates[2] + c[3]
        )
    elif len(c) == 3:
        val = c[0] * coordinates[0] + c[1] * coordinates[1] + c[2]
    else:
        val = c[0] * coordinates[0] + c[1]

    return val


def poly2col(polynomial: np.ndarray) -> list:
    """
    Returns the coefficients (columns) of a polynomial in the form of a list.

    Parameters
    ----------
        polynomial (np.ndarray): Coefficients, i.e., the ones obtained from interpolate_P1. The
            expected shape is: rows x num_lagrangian_nodes.

    Returns
    -------
        List
            Coefficients stored in column-wise format.

    """
    rows = polynomial.shape[0]
    cols = polynomial.shape[1]
    coeff_list = []

    for col in range(cols):
        coeff_list.append(polynomial[:, col].reshape(rows, 1))

    return coeff_list
