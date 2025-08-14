"""
This module contains functional tests on the solution and errors for the third example.
"""
from __future__ import annotations


import numpy as np
import porepy as pp

import mdnme
from mdnme.examples.benchmark_3d_case_3.model import SmallFeaturesModel, solid_constants


def test_example_3_with_refinement_level_0() -> None:
    """Test whether we obtain the pressure norm and majorant for the third example."""

    # Declare model parameters
    params = {
        "material_constants": {"solid": solid_constants},
        "supress_outputs_for_tests": [],  # Supress outputs for tests
    }

    # Run the model
    model = SmallFeaturesModel(params)
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
    assert np.isclose(norm_p_num, 13.337947636529996, atol=1e0)

    # Check majorant
    majorant = mdnme.get_majorant(mdg)
    assert np.isclose(majorant, 17.621880418006178, atol=1e0)
