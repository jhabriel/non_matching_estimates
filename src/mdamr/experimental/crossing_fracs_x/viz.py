import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from data_saving import TwoCrossingSaveData
from porepy.utils.examples_utils import VerificationUtils


class TwoCrossingUtils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[TwoCrossingSaveData]
    """List of TwoCrossingSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_matrix_pressure()
        self._plot_fractures_pressure()
        # self._plot_interface_fluxes()
        # self._plot_fracture_fluxes()

    def _plot_matrix_pressure(self) -> None:
        """Plots exact and numerical pressures in the matrix."""
        sd_matrix = self.mdg.subdomains()[0]
        p_num = self.results[-1].approx_p_2d
        p_ex = self.results[-1].exact_p_2d
        pp.plot_grid(
            sd_matrix, p_ex, plot_2d=True, linewidth=0, title="Matrix pressure (Exact)"
        )
        pp.plot_grid(
            sd_matrix, p_num, plot_2d=True, linewidth=0, title="Matrix pressure (MPFA)"
        )

    def _plot_fractures_pressure(self):
        """Plots exact and numerical pressures in the fracture."""

        sd_1d_west = self.mdg.subdomains()[1]
        cc = sd_1d_west.cell_centers
        p_num = self.results[-1].approx_p_1d_west
        p_ex = self.results[-1].exact_p_1d_west
        plt.plot(p_ex, cc[0], label="Exact", linewidth=3, alpha=0.5)
        plt.plot(p_num, cc[0], label="MPFA", marker=".", markersize=5, linewidth=0)
        plt.xlabel("Fracture pressure")
        plt.ylabel("x-coordinate")
        plt.legend()
        plt.show()

    #
    # def _plot_interface_fluxes(self):
    #     """Plots exact and numerical interface fluxes."""
    #     intf = self.mdg.interfaces()[0]
    #     cc = intf.cell_centers
    #     lmbda_num = self.results[-1].approx_intf_flux
    #     lmbda_ex = self.results[-1].exact_intf_flux
    #     plt.plot(lmbda_ex, cc[1], label="Exact", linewidth=3, alpha=0.5)
    #     plt.plot(lmbda_num, cc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
    #     plt.xlabel("Interface flux")
    #     plt.ylabel("y-coordinate")
    #     plt.legend()
    #     plt.show()
    #
    # def _plot_fracture_fluxes(self):
    #     """Plots exact and numerical fracture fluxes."""
    #     sd_frac = self.mdg.subdomains()[1]
    #     fc = sd_frac.face_centers
    #     q_num = self.results[-1].approx_frac_flux
    #     q_ex = self.results[-1].exact_frac_flux
    #     plt.plot(q_ex, fc[1], label="Exact", linewidth=3, alpha=0.5)
    #     plt.plot(q_num, fc[1], label="MPFA", marker=".", markersize=5, linewidth=0)
    #     plt.xlabel("Fracture Darcy flux")
    #     plt.ylabel("y-coordinate")
    #     plt.legend()
    #     plt.show()
