"""
Module to plot the majorants and the efficiency indices for example 1 from the paper.
"""

import matplotlib.pyplot as plt
from porepy.utils.txt_io import read_data_from_txt

# %% Read data
data = read_data_from_txt("error_analysis.txt")

dofs = data["dofs"]
true_errors = data["true_error"]
majorants = data["majorant"]
eff_indices = data["eff_index"]


# %% Plot data
fig, axs = plt.subplots(2, 1, sharex=True)

color = "tab:red"
axs[0].set_ylabel("Majorant")
axs[0].loglog(dofs, majorants, color=color, label="uniform", linewidth=2)
axs[0].legend()

color = "tab:blue"
axs[1].set_xlabel("Global degrees of freedom")
axs[1].set_ylabel("Efficiency index")
axs[1].semilogx(dofs, eff_indices, color=color, label="uniform", linewidth=2)
axs[1].legend()

fig.tight_layout()
plt.show()
