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


from modelzoo.common.pytorch.run_utils import run
from modelzoo.fc_mnist.pytorch.utils import set_defaults

from benchmarking.model import BenchmarkModel
from benchmarking.data import get_eval_dataloader

def main():
    run(BenchmarkModel, eval_data_fn = get_eval_dataloader, default_params_fn = set_defaults)


if __name__ == '__main__':
    main()
