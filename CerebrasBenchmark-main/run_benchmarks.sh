#!/bin/bash

mkdir -p outputs

python3 benchmarking/configs/make_params.py

for pfile in benchmarking/configs/generated/*.yaml; do
    strip_pref=${pfile##*/params_}
    name=${strip_pref%\.yaml}

    echo "running $name"

    python3 -m benchmarking.run CSX weight_streaming \
    --job_labels name=cs_benchmark \
    --params $pfile \
    --mode eval \
    --mount_dirs /home/ /software \
    --python_paths $(pwd) $(pwd)/modelzoo \
    --model_dir outputs_nan/model_$name

done