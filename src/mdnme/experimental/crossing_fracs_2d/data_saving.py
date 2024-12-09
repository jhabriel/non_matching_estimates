from dataclasses import dataclass
from typing import Callable

import numpy as np
import porepy as pp
from exact_sol import TwoCrossingExactSolution
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.viz.data_saving_model_mixin import VerificationDataSaving

number = pp.number


@dataclass
class TwoCrossingSaveData:
    """Data class to save relevant results from the verification setup."""

    # Approximate solutions
    approx_p_2d: np.ndarray
    # approx_p_1d_west: np.ndarray
    # approx_p_1d_east: np.ndarray
    # approx_p_1d_north: np.ndarray
    # approx_p_1d_south: np.ndarray
    # approx_p_0d: np.ndarray

    # approx_q_2d: np.ndarray
    # approx_q_1d_west: np.ndarray
    # approx_q_1d_east: np.ndarray
    # approx_q_1d_south: np.ndarray
    # approx_q_1d_north: np.ndarray
    #
    # approx_lmbda_1d_west: np.ndarray
    # approx_lmbda_1d_east: np.ndarray
    # approx_lmbda_1d_south: np.ndarray
    # approx_lmbda_1d_north: np.ndarray
    # approx_lmbda_0d_west: np.ndarray
    # approx_lmbda_0d_east: np.ndarray
    # approx_lmbda_0d_south: np.ndarray
    # approx_lmbda_0d_north: np.ndarray

    # Exact solutions
    exact_p_2d: np.ndarray
    # exact_p_1d_west: np.ndarray
    # exact_p_1d_east: np.ndarray
    # exact_p_1d_south: np.ndarray
    # exact_p_1d_north: np.ndarray
    # exact_p_0d: np.ndarray

    # exact_q_2d: np.ndarray
    # exact_q_1d_west: np.ndarray
    # exact_q_1d_east: np.ndarray
    # exact_q_1d_south: np.ndarray
    # exact_q_1d_north: np.ndarray
    #
    # exact_lmbda_1d_west: np.ndarray
    # exact_lmbda_1d_east: np.ndarray
    # exact_lmbda_1d_south: np.ndarray
    # exact_lmbda_1d_north: np.ndarray
    # exact_lmbda_0d_west: np.ndarray
    # exact_lmbda_0d_east: np.ndarray
    # exact_lmbda_0d_south: np.ndarray
    # exact_lmbda_0d_north: np.ndarray

    # L2-relative error
    error_p_2d: number
    # error_p_1d_west: number
    # error_p_1d_east: number
    # error_p_1d_south: number
    # error_p_1d_north: number
    # error_p_0d: number

    # error_q_2d: number
    # error_q_1d_west: number
    # error_q_1d_east: number
    # error_q_1d_south: number
    # error_q_1d_north: number
    #
    # error_lmbda_1d_west: number
    # error_lmbda_1d_east: number
    # error_lmbda_1d_south: number
    # error_lmbda_1d_north: number
    # error_lmbda_0d_west: number
    # error_lmbda_0d_east: number
    # error_lmbda_0d_south: number
    # error_lmbda_0d_north: number


class TwoCrossingDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    # TODO: Add proper docstrings to the typings
    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    equation_system: pp.EquationSystem
    exact_sol: TwoCrossingExactSolution
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    mdg: pp.MixedDimensionalGrid

    def collect_data(self) -> TwoCrossingSaveData:
        """Collect data from the verification setup.

        Returns:
            TwoCrossingSaveData object containing the results of the verification.

        """

        # Retrieve subdomains and interfaces
        # Numbering of lower-dimensional subdomains and interfaces follows
        # the rule: West -> East -> South -> North
        sd_2d: pp.Grid = self.mdg.subdomains()[0]
        sd_1d_west: pp.Grid = self.mdg.subdomains()[1]
        sd_1d_east: pp.Grid = self.mdg.subdomains()[2]
        sd_1d_south: pp.Grid = self.mdg.subdomains()[3]
        sd_1d_north: pp.Grid = self.mdg.subdomains()[4]
        sd_0d: pp.Grid = self.mdg.subdomains()[5]
        intf_1d_west: pp.MortarGrid = self.mdg.interfaces()[0]
        intf_1d_east: pp.MortarGrid = self.mdg.interfaces()[1]
        intf_1d_south: pp.MortarGrid = self.mdg.interfaces()[2]
        intf_1d_north: pp.MortarGrid = self.mdg.interfaces()[3]
        intf_0d_west: pp.MortarGrid = self.mdg.interfaces()[4]
        intf_0d_east: pp.MortarGrid = self.mdg.interfaces()[5]
        intf_0d_south: pp.MortarGrid = self.mdg.interfaces()[6]
        intf_0d_north: pp.MortarGrid = self.mdg.interfaces()[7]

        # Retrieve exact solution object
        exact_sol: TwoCrossingExactSolution = self.exact_sol

        # Collect data

        # Pressure, 2d subdomain
        exact_p_2d = exact_sol.p_2d(sd_2d)
        approx_p_2d = self.pressure([sd_2d]).evaluate(self.equation_system).val
        error_p_2d = ConvergenceAnalysis.l2_error(
            grid=sd_2d,
            true_array=exact_p_2d,
            approx_array=approx_p_2d,
            is_scalar=True,
            is_cc=True,
            relative=True,
        )

        # # Pressure, 1d subdomain, West
        # exact_p_1d_west = exact_sol.p_1d_west(sd_1d_west)
        # approx_p_1d_west = (
        #     self.pressure([sd_1d_west]).evaluate(self.equation_system).val
        # )
        # error_p_1d_west = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_west,
        #     true_array=exact_p_1d_west,
        #     approx_array=approx_p_1d_west,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Pressure, 1d subdomain, East
        # exact_p_1d_east = exact_sol.p_1d_east(sd_1d_east)
        # approx_p_1d_east = (
        #     self.pressure([sd_1d_east]).evaluate(self.equation_system).val
        # )
        # error_p_1d_east = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_east,
        #     true_array=exact_p_1d_east,
        #     approx_array=approx_p_1d_east,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Pressure, 1d subdomain, South
        # exact_p_1d_south = exact_sol.p_1d_south(sd_1d_south)
        # approx_p_1d_south = (
        #     self.pressure([sd_1d_south]).evaluate(self.equation_system).val
        # )
        # error_p_1d_south = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_south,
        #     true_array=exact_p_1d_south,
        #     approx_array=approx_p_1d_south,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Pressure, 1d subdomain, North
        # exact_p_1d_north = exact_sol.p_1d_north(sd_1d_north)
        # approx_p_1d_north = (
        #     self.pressure([sd_1d_north]).evaluate(self.equation_system).val
        # )
        # error_p_1d_north = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_north,
        #     true_array=exact_p_1d_north,
        #     approx_array=approx_p_1d_north,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Pressure, 0d subdomain
        # exact_p_0d = exact_sol.p_0d(sd_0d)
        # approx_p_0d = self.pressure([sd_0d]).evaluate(self.equation_system).val
        # error_p_0d = ConvergenceAnalysis.l2_error(
        #     grid=sd_0d,
        #     true_array=exact_p_0d,
        #     approx_array=approx_p_0d,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )

        # Darcy flux, 2d subdomain
        # exact_q_2d = exact_sol.q_2d(sd_2d)
        # approx_q_2d = self.darcy_flux([sd_2d]).evaluate(self.equation_system).val
        # error_q_2d = ConvergenceAnalysis.l2_error(
        #     grid=sd_2d,
        #     true_array=exact_q_2d,
        #     approx_array=approx_q_2d,
        #     is_scalar=True,
        #     is_cc=False,
        #     relative=True,
        # )

        # # Darcy flux, 1d subdomain, West
        # exact_q_1d_west = exact_sol.q_1d_west(sd_1d_west)
        # approx_q_1d_west = (
        #     self.darcy_flux([sd_1d_west]).evaluate(self.equation_system).val
        # )
        # error_q_1d_west = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_west,
        #     true_array=exact_q_1d_west,
        #     approx_array=approx_q_1d_west,
        #     is_scalar=True,
        #     is_cc=False,
        #     relative=True,
        # )
        #
        # # Darcy flux, 1d subdomain, East
        # exact_q_1d_east = exact_sol.q_1d_east(sd_1d_east)
        # approx_q_1d_east = (
        #     self.darcy_flux([sd_1d_east]).evaluate(self.equation_system).val
        # )
        # error_q_1d_east = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_east,
        #     true_array=exact_q_1d_east,
        #     approx_array=approx_q_1d_east,
        #     is_scalar=True,
        #     is_cc=False,
        #     relative=True,
        # )
        #
        # # Darcy flux, 1d subdomain, South
        # exact_q_1d_south = exact_sol.q_1d_south(sd_1d_south)
        # approx_q_1d_south = (
        #     self.darcy_flux([sd_1d_south]).evaluate(self.equation_system).val
        # )
        # error_q_1d_south = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_south,
        #     true_array=exact_q_1d_south,
        #     approx_array=approx_q_1d_south,
        #     is_scalar=True,
        #     is_cc=False,
        #     relative=True,
        # )
        #
        # # Darcy flux, 1d subdomain, North
        # exact_q_1d_north = exact_sol.q_1d_north(sd_1d_north)
        # approx_q_1d_north = (
        #     self.darcy_flux([sd_1d_north]).evaluate(self.equation_system).val
        # )
        # error_q_1d_north = ConvergenceAnalysis.l2_error(
        #     grid=sd_1d_north,
        #     true_array=exact_q_1d_north,
        #     approx_array=approx_q_1d_north,
        #     is_scalar=True,
        #     is_cc=False,
        #     relative=True,
        # )
        #
        # # Interface flux, 1d interface, West
        # exact_lmbda_1d_west = exact_sol.lmbda_1d_west(intf_1d_west)
        # approx_lmbda_1d_west = (
        #     self.interface_darcy_flux([intf_1d_west]).evaluate(self.equation_system).val
        # )
        # error_lmbda_1d_west = ConvergenceAnalysis.l2_error(
        #     grid=intf_1d_west,
        #     true_array=exact_lmbda_1d_west,
        #     approx_array=approx_lmbda_1d_west,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 1d interface, East
        # exact_lmbda_1d_east = exact_sol.lmbda_1d_east(intf_1d_east)
        # approx_lmbda_1d_east = (
        #     self.interface_darcy_flux([intf_1d_east]).evaluate(self.equation_system).val
        # )
        # error_lmbda_1d_east = ConvergenceAnalysis.l2_error(
        #     grid=intf_1d_east,
        #     true_array=exact_lmbda_1d_east,
        #     approx_array=approx_lmbda_1d_east,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 1d interface, South
        # exact_lmbda_1d_south = exact_sol.lmbda_1d_south(intf_1d_south)
        # approx_lmbda_1d_south = (
        #     self.interface_darcy_flux([intf_1d_south])
        #     .evaluate(self.equation_system)
        #     .val
        # )
        # error_lmbda_1d_south = ConvergenceAnalysis.l2_error(
        #     grid=intf_1d_south,
        #     true_array=exact_lmbda_1d_south,
        #     approx_array=approx_lmbda_1d_south,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 1d interface, North
        # exact_lmbda_1d_north = exact_sol.lmbda_1d_north(intf_1d_north)
        # approx_lmbda_1d_north = (
        #     self.interface_darcy_flux([intf_1d_north])
        #     .evaluate(self.equation_system)
        #     .val
        # )
        # error_lmbda_1d_north = ConvergenceAnalysis.l2_error(
        #     grid=intf_1d_north,
        #     true_array=exact_lmbda_1d_north,
        #     approx_array=approx_lmbda_1d_north,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 0d interface, West
        # exact_lmbda_0d_west = exact_sol.lmbda_0d_west(intf_0d_west)
        # approx_lmbda_0d_west = (
        #     self.interface_darcy_flux([intf_0d_west]).evaluate(self.equation_system).val
        # )
        # error_lmbda_0d_west = ConvergenceAnalysis.l2_error(
        #     grid=intf_0d_west,
        #     true_array=exact_lmbda_0d_west,
        #     approx_array=approx_lmbda_0d_west,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 0d interface, East
        # exact_lmbda_0d_east = exact_sol.lmbda_0d_east(intf_0d_east)
        # approx_lmbda_0d_east = (
        #     self.interface_darcy_flux([intf_0d_east]).evaluate(self.equation_system).val
        # )
        # error_lmbda_0d_east = ConvergenceAnalysis.l2_error(
        #     grid=intf_0d_east,
        #     true_array=exact_lmbda_0d_east,
        #     approx_array=approx_lmbda_0d_east,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 0d interface, South
        # exact_lmbda_0d_south = exact_sol.lmbda_0d_south(intf_0d_south)
        # approx_lmbda_0d_south = (
        #     self.interface_darcy_flux([intf_0d_south])
        #     .evaluate(self.equation_system)
        #     .val
        # )
        # error_lmbda_0d_south = ConvergenceAnalysis.l2_error(
        #     grid=intf_0d_south,
        #     true_array=exact_lmbda_0d_south,
        #     approx_array=approx_lmbda_0d_south,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )
        #
        # # Interface flux, 0d interface, North
        # exact_lmbda_0d_north = exact_sol.lmbda_0d_north(intf_0d_north)
        # approx_lmbda_0d_north = (
        #     self.interface_darcy_flux([intf_0d_north])
        #     .evaluate(self.equation_system)
        #     .val
        # )
        # error_lmbda_0d_north = ConvergenceAnalysis.l2_error(
        #     grid=intf_0d_north,
        #     true_array=exact_lmbda_0d_north,
        #     approx_array=approx_lmbda_0d_north,
        #     is_scalar=True,
        #     is_cc=True,
        #     relative=True,
        # )

        # Store collected data in data class
        collected_data = TwoCrossingSaveData(
            approx_p_2d=approx_p_2d,
            # approx_p_1d_west=approx_p_1d_west,
            # approx_p_1d_east=approx_p_1d_east,
            # approx_p_1d_south=approx_p_1d_south,
            # approx_p_1d_north=approx_p_1d_north,
            # approx_p_0d=approx_p_0d,
            # approx_q_2d=approx_q_2d,
            # approx_q_1d_west=approx_q_1d_west,
            # approx_q_1d_east=approx_q_1d_east,
            # approx_q_1d_south=approx_q_1d_south,
            # approx_q_1d_north=approx_q_1d_north,
            # approx_lmbda_1d_west=approx_lmbda_1d_west,
            # approx_lmbda_1d_east=approx_lmbda_1d_east,
            # approx_lmbda_1d_south=approx_lmbda_1d_south,
            # approx_lmbda_1d_north=approx_lmbda_1d_north,
            # approx_lmbda_0d_west=approx_lmbda_0d_west,\
            # approx_lmbda_0d_east=approx_lmbda_0d_east,
            # approx_lmbda_0d_south=approx_lmbda_0d_south,
            # approx_lmbda_0d_north=approx_lmbda_0d_north,
            exact_p_2d=exact_p_2d,
            # exact_p_1d_west=exact_p_1d_west,
            # exact_p_1d_east=exact_p_1d_east,
            # exact_p_1d_south=exact_p_1d_south,
            # exact_p_1d_north=exact_p_1d_north,
            # exact_p_0d=exact_p_0d,
            # exact_q_2d=exact_q_2d,
            # exact_q_1d_west=exact_q_1d_west,
            # exact_q_1d_east=exact_q_1d_east,
            # exact_q_1d_south=exact_q_1d_south,
            # exact_q_1d_north=exact_q_1d_north,
            # exact_lmbda_1d_west=exact_lmbda_1d_west,
            # exact_lmbda_1d_east=exact_lmbda_1d_east,
            # exact_lmbda_1d_south=exact_lmbda_1d_south,
            # exact_lmbda_1d_north=exact_lmbda_1d_north,
            # exact_lmbda_0d_west=exact_lmbda_0d_west,
            # exact_lmbda_0d_east=exact_lmbda_0d_east,
            # exact_lmbda_0d_south=exact_lmbda_0d_south,
            # exact_lmbda_0d_north=exact_lmbda_0d_north,
            error_p_2d=error_p_2d,
            # error_p_1d_west=error_p_1d_west,
            # error_p_1d_east=error_p_1d_east,
            # error_p_1d_south=error_p_1d_south,
            # error_p_1d_north=error_p_1d_north,
            # error_p_0d=error_p_0d,
            # #error_q_2d=error_q_2d,
            # error_q_1d_west=error_q_1d_west,
            # error_q_1d_east=error_q_1d_east,
            # error_q_1d_south=error_q_1d_south,
            # error_q_1d_north=error_q_1d_north,
            # error_lmbda_1d_west=error_lmbda_1d_west,
            # error_lmbda_1d_east=error_lmbda_1d_east,
            # error_lmbda_1d_south=error_lmbda_1d_south,
            # error_lmbda_1d_north=error_lmbda_1d_north,
            # error_lmbda_0d_west=error_lmbda_0d_west,
            # error_lmbda_0d_east=error_lmbda_0d_east,
            # error_lmbda_0d_south=error_lmbda_0d_south,
            # error_lmbda_0d_north=error_lmbda_0d_north,
        )

        return collected_data
