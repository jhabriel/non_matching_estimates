from typing import Callable, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

def _nnz_per_axis(A: sps.spmatrix, axis: int, tol: float) -> np.ndarray:
    """Count 'significant' nonzeros (>tol) per row (axis=0) or column (axis=1)."""
    if not sps.isspmatrix(A):
        A = sps.coo_matrix(A)
    else:
        A = A.tocoo()

    mask = np.abs(A.data) > tol
    rows = A.row[mask]
    cols = A.col[mask]

    if axis == 0:  # per-row
        return np.bincount(rows, minlength=A.shape[0])
    elif axis == 1:  # per-col
        return np.bincount(cols, minlength=A.shape[1])
    else:
        raise ValueError("axis must be 0 (rows) or 1 (cols)")


def is_nonmatching(intf: pp.MortarGrid, tol: float = 1e-12, mode: str = "strict") -> bool:
    """
    Heuristic detector for (non-)matching mortar interfaces.

    Returns True if the interface behaves as *non-matching*.

    Logic:
      - For a matching 3D–2D interface (2D mortar):
          * Each mortar cell (row) should get exactly ONE contribution from
            the primary side (a single high-dim face)  -> row nnz(primary)=1
          * Each mortar cell (row) should get exactly ONE contribution from
            the secondary side (one low-dim cell)      -> row nnz(secondary)=1
        These two are the essential checks.

      - In 'strict' mode we also enforce:
          * Each primary face maps to exactly ONE mortar cell (on its side)
            -> col nnz(primary)=1
          * Each secondary cell appears exactly once per mortar side, i.e.
            total column nnz across both sides equals intf.num_sides()
            -> col nnz(secondary)=intf.num_sides()

    Notes:
      - tol filters tiny numerical noise in mapping weights.
      - Works for 1D mortars analogously.
    """
    # Quick exits for degenerate dimensions
    if intf.dim == 0:
        return False

    Pm = intf.primary_to_mortar_avg()     # shape: (n_mortar, n_primary_faces)
    Sm = intf.secondary_to_mortar_avg()   # shape: (n_mortar, n_secondary_cells)

    # Essential row-wise checks
    rnnz_P = _nnz_per_axis(Pm, axis=0, tol=tol)  # per mortar cell
    rnnz_S = _nnz_per_axis(Sm, axis=0, tol=tol)

    nonmatching = (np.any(rnnz_P != 1) or np.any(rnnz_S != 1))

    if mode.lower() == "strict":
        # Column-wise uniqueness (see docstring)
        cnnz_P = _nnz_per_axis(Pm, axis=1, tol=tol)  # per primary face
        cnnz_S = _nnz_per_axis(Sm, axis=1, tol=tol)  # per secondary cell
        expected_S = intf.num_sides()  # 1 or 2
        nonmatching = (
            nonmatching or
            np.any(cnnz_P != 1) or
            np.any(cnnz_S != expected_S)
        )

    return bool(nonmatching)


class ErrorEstimatesSaveData:
    """Mixin class that saves data needed for error estimation."""

    darcy_keyword: str
    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    equation_system: pp.EquationSystem
    normal_permeability: Callable
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    mdg: pp.MixedDimensionalGrid
    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    aperture: Callable[[list[pp.Grid]], pp.ad.Operator]
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    permeability_tensor: Callable[[pp.Grid], pp.SecondOrderTensor]

    def error_estimates_data_saving(self) -> None:
        """Save data"""
        eqsys = self.equation_system

        # Save data for subdomain variables
        for sd, d in self.mdg.subdomains(return_data=True):
            # Create key if it does not exist
            if d.get("estimates") is None:
                d["estimates"] = {}

            # Save subdomain MPFA pressure
            d["estimates"]["fv_sd_pressure"] = self.pressure([sd]).value(eqsys)

            # Save subdomain MPFA Darcy fluxes
            d["estimates"]["fv_sd_flux"] = self.darcy_flux([sd]).value(eqsys)

        # Save data for interface variables
        for intf, d in self.mdg.interfaces(return_data=True):
            # Create key if it does not exist
            if d.get("estimates") is None:
                d["estimates"] = {}

            # Save interface MPFA fluxes
            d["estimates"]["fv_intf_flux"] = self.interface_darcy_flux([intf]).value(
                eqsys
            )

            # Save effective normal permeability
            d[pp.PARAMETERS]["flow"][
                "effective_permeability"
            ] = self.effective_normal_permeability([intf]).value(eqsys)

        # Save sources from interface fluid fluxes
        for sd, d in self.mdg.subdomains(return_data=True):
            self._internal_sources_from_interfaces(sd, d)

    def _internal_sources_from_interfaces(self, sd: pp.Grid, d: dict) -> None:
        """
        Compute the internal source contributions from (higher-dimensional) neighboring
        interfaces.

        :param sd:
        :return:
        """

        # Handle the case where we have only one subdomain
        if self.mdg.num_subdomains() == 1:
            d["estimates"]["sources_from_intf"] = np.zeros(sd.num_cells).reshape(
                sd.num_cells, 1
            )

        # Initialize array of internal sources
        internal_source = np.zeros(sd.num_cells)

        # Retrieve list of neighboring higher-dimensional subdomains
        sd_highs = self.mdg.neighboring_subdomains(sd, only_higher=True)

        # Loop through all the higher-dimensional adjacent interfaces w.r.t the
        # lower-dimensional subdomain to map the interface fluxes to internal
        # source terms
        for sd_high in sd_highs:
            intf = self.mdg.subdomain_pair_to_interface((sd, sd_high))
            data_intf = self.mdg.interface_data(intf)
            fv_intf_vel = data_intf["estimates"]["fv_intf_flux"] / intf.cell_volumes
            # Obtain source term contribution associated to the neighboring interface
            internal_source += intf.mortar_to_secondary_int() * fv_intf_vel

        d["estimates"]["sources_from_intf"] = internal_source.reshape(sd.num_cells, 1)

    def effective_normal_permeability(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """
        Computes the effective normal permeability, see Eq. 6b from [1].

        The effective normal permeability is the scalar that multiplies the pressure
        jump in the continuous interface law.

        Parameters:
            interfaces: List of pp.MortarGrid
                List of interface grids.

        Returns:
            Wrapped ad operator containing the effective normal permeabilities for the
            given list of interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg()
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        effective_normal_permeability = (
            self.specific_volume(interfaces)
            * self.normal_permeability(interfaces)
            * normal_gradient
        )
        effective_normal_permeability.set_name("effective_normal_permeability")

        return effective_normal_permeability
