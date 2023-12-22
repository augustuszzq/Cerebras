"""
    Duplicates params.yaml to have different sparsities
"""

import yaml
import os
import copy
import itertools

# 0.0 is fully dense, 1.0 means all zeroes.
weight_sparsities = [-1] + [0.2 * i for i in range(1, 6)]
sparse_modes = ["col", "row"]

parameters = list(itertools.product(weight_sparsities, sparse_modes))


root = os.path.dirname(__file__)

# read in the data to copy
with open(os.path.join(root, "params.yaml")) as param_file:
    params = yaml.safe_load(param_file)

# Duplicate it for each sparsity and a no-op
names = [
    f"params_{mode}_{'no_op' if sparsity < 0 else '{:.2f}'.format(sparsity)}" for sparsity, mode in parameters
]

for name, (sparsity, mode) in zip(names, parameters):
    param_copy = copy.deepcopy(params)
    sparse_value = min(max(sparsity, 0.001), 0.99)
    param_copy["model"]["sparsity"] = sparse_value
    param_copy["model"]["do_op"] = sparsity >= 0
    param_copy["model"]["sparse_mode"] = mode

    # We may want to be running dense ops without sparsity section
    if param_copy.get("sparsity", False):
        param_copy["sparsity"]["sparsity"] = sparse_value
    else:
        name += "_dense"

    with open(
        os.path.join(root, "generated", f"{name}.yaml"), "w"
    ) as param_file:
        yaml.dump(param_copy, param_file)
