"""Module for running the third numerical example"""
import mdnme
import porepy as pp
import numpy as np

from mdnme.examples.bit_example_3.model import SmallFeaturesModel, solid_constants

#REFINEMENT_LEVELS = [0, 1, 2, 3]
REFINEMENT_LEVELS = [0]

for lvl in REFINEMENT_LEVELS:

    # Setup the model and solve using MPFA
    params = {
        "material_constants": {"solid": solid_constants},
        "refinement_level": lvl,
        "non_matching": True,
    }
    model = SmallFeaturesModel(params)

    print(f"Solving model for refinement level {lvl}.")
    pp.run_time_dependent_model(model, params)
    print(f"Done solving model.")

    # Estimate the errors
    print(f"Estimating errors for refinement level {lvl}.")
    mdnme.estimate_errors(mdg=model.mdg)
    print(f"Done estimating errors.")


