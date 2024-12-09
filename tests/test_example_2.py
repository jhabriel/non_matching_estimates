"""
This module contains functional tests on the solution and errors for the second example.
"""
from __future__ import annotations

import numpy as np

import porepy as pp

import mdnme
from mdnme.examples.example_2.model import ComplexNetworkModel, solid_constants


def test_example_2_with_mesh_size_005() -> None:
    """Test whether we obtain the pressure norma and majorant for the second example."""

    # Declare model parameters
    params = {
        "material_constants": {"solid": solid_constants},
        "grid_type": "simplex",
        "meshing_arguments": {"cell_size": 0.05},
    }

    # Run the model
    model = ComplexNetworkModel(params)
    pp.run_time_dependent_model(model, {})
    mdg = model.mdg

    # Estimate errors
    mdnme.estimate_errors(
        mdg,
        pressure_reconstruction_method="keilegavlen_p1",
    )

    # Check pressure norm
    p_num = model.pressure(mdg.subdomains()).value(model.equation_system)
    norm_p_num = np.linalg.norm(p_num)
    assert np.isclose(norm_p_num, 107.17457971877116)

    # Check majorant
    majorant = mdnme.get_majorant(mdg)
    assert np.isclose(majorant, 866.5296233755787)
