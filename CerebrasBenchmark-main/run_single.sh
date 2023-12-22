#!/bin/bash

mkdir -p outputs

python3 -m benchmarking.run CSX weight_streaming \
    --job_labels name=cs_benchmark \
    --params benchmarking/configs/params.yaml \
    --mode eval \
    --mount_dirs /home/ /software \
    --python_paths $(pwd) $(pwd)/modelzoo \
    --model_dir outputs/model_defaults