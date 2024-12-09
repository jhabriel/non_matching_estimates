from typing import Callable, Union

import numpy as np
import porepy as pp


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
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )
        effective_normal_permeability = (
            self.specific_volume(interfaces)
            * self.normal_permeability(interfaces)
            * normal_gradient
        )
        effective_normal_permeability.set_name("effective_normal_permeability")

        return effective_normal_permeability
