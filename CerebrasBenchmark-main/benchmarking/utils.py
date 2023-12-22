# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from functools import reduce

def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    params["train_input"]["shuffle"] = params["train_input"].get(
        "shuffle", True
    )
    params["eval_input"]["shuffle"] = params["eval_input"].get("shuffle", False)

    params["model"]["to_float16"] = params["model"].get("to_float16", False)
    params["model"]["use_bfloat16"] = params["model"].get("use_bfloat16", False)

prod = lambda tup: reduce(lambda a, b: a*b, tup)


def set_zeros_(tensor: torch.Tensor, sparsity: float, fill_val = 0.0, mode = "random"):
    """
    Randomly sets the given proportion of entries of a tensor to zero in-place.

    Allowed modes: "random", "row" (first few rows), "col" (first few columns).

    Tensor is assumed to be 2d
    """
    n_entries = prod(tensor.shape)
    n_zeros = round(sparsity * n_entries)
    n_nonzero = n_entries - n_zeros

    # Create the mask depending on the mode
    if mode == "random":
        # Sparsify random tensors
        sparsity_mask = torch.tensor(np.random.choice([True, False], size = tensor.shape, p = (sparsity, 1-sparsity)), dtype = bool)
    elif mode in ("row", "col"):
        # Sparsify the first rows
        sparsity_mask = np.array([True] * n_zeros + [False] * n_nonzero)

        if mode == "row":
            sparsity_mask = sparsity_mask.reshape(tensor.shape)
        if mode =="col":
            sparsity_mask = sparsity_mask.reshape((tensor.shape[1], tensor.shape[0]))
            sparsity_mask = sparsity_mask.transpose()

        sparsity_mask = torch.tensor(sparsity_mask, dtype = bool)

    else:
        raise Exception(f"Sparsity mode {mode} not recognized")

    # Perform the fill
    tensor.masked_fill_(sparsity_mask, fill_val)