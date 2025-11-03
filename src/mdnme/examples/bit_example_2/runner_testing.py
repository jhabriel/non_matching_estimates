"""Runner"""

import numpy as np
import porepy as pp
import mdnme

from mdnme.examples.bit_example_2.flow_benchmark_3d_case_2 import (
    FlowBenchmark3dCase2Model,
    solid_constants_conductive,
    solid_constants_blocking
)

# Set parameters dictionary
params = {
    "material_constants": {"solid": solid_constants_conductive},
    "refinement_level": 0,
    "non_matching": False,
}
model = FlowBenchmark3dCase2Model(params)
pp.run_time_dependent_model(model, {})
