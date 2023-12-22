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

from copy import deepcopy

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.common.pytorch.summaries import scalar_summary
from modelzoo.common.pytorch.sparsity.torch_sparsify import sparsify_pytorch_model
from benchmarking.utils import set_zeros_


class DenseBenchmarkModel(nn.Module):
    def __init__(self, model_params: dict):
        super().__init__()

        # check existence of important parameters
        assert (
            "dim_in" in model_params and "dim_out" in model_params
        ), "Must specify matrix dimensions"

        sparsity = float(model_params.get("sparsity", 0.0))
        sparse_val = float(model_params["sparse_value"])
        sparse_mode = model_params["sparse_mode"]

        self.do_op = model_params.get("do_op", sparsity <= 1)

        mat_shape = (model_params["dim_in"], model_params["dim_out"])

        
        # Initialize multiplication operation
        self.lin_layer = nn.Linear(*mat_shape, bias=False)

        with torch.no_grad():
            # normalize by size to prevent getting numerical issues from successive multiplications
            self.mat_entry = 1.0 / mat_shape[0]
            self.lin_layer.weight.fill_(self.mat_entry)  
            # Not manually setting sparsity here, using the built-in sparsifier
            set_zeros_(self.lin_layer.weight, sparsity, fill_val = sparse_val, mode = sparse_mode)

    def forward(self, x):
        return self.lin_layer(x)


class BenchmarkModel(nn.Module):
    """
    Mirrors the structure of the ModelZoo MNIST example to wrap the pytorch
    model for execution on Cerebras
    """

    def __init__(self, params):
        super().__init__()

        model_params = deepcopy(params["model"])

        # The parameters effect how many distict multiplications we do and how many times they
        # are repeated in sequence.
        # As a control group, we also run some models that skip all the heavy operations
        self.n_repeats = model_params.get("n_repeats", 1)
        self.n_models = model_params.get("n_matrices", 1)
        self.do_op = model_params.get("do_op", True)

        # Create and register models
        self.models = [self.build_model(model_params) for _ in range(self.n_models)]
        for i, model in enumerate(self.models):
            self.add_module(f"mat_mul_{i}", model)

        # Nonlinearity to prevent compiler from combining linear operations
        self.nonlin = nn.Sigmoid()

        # Any loss theoretically works
        self.loss = torch.nn.MSELoss()

    def build_model(self, model_params):
        dtype = (
            half_dtype_instance.half_dtype
            if model_params["to_float16"]
            else torch.float32
        )

        model = DenseBenchmarkModel(model_params)
        model.to(dtype)

        return model

    def __call__(self, data):
        inputs, labels = data

        # Need something to compare to in the loss function
        orig_input = inputs.clone()

        if self.do_op:
            for rep in range(self.n_repeats):
                for i, model in enumerate(self.models):
                    inputs = model(inputs)
                    # inputs = self.nonlin(inputs)
                
        return self.loss(inputs, orig_input)
