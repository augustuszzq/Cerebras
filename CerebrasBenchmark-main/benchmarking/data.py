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

import torch
import torch.utils.data

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.utils import get_input_dtype

def get_eval_dataloader(params):
    # read in the parameters
    input_params = params["eval_input"]
    use_cs = cm.use_cs() or cm.is_appliance()

    batch_size = input_params["batch_size"]

    n_samples = input_params["n_samples"]
    dim = input_params["dim"]

    to_float16 = input_params.get("to_float16", True)
    dtype = get_input_dtype(to_float16)

    dataset_shape = (n_samples, dim)

    inputs = torch.rand(dataset_shape)
    # apply sparsity
    # inputs = set_zeros(inputs, sparsity)
    inputs = inputs.to(dtype)

    dataset = torch.utils.data.TensorDataset(
        inputs, torch.ones(n_samples, dtype=torch.int32)
    )

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
