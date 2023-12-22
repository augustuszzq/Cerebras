# Cerebras Matrix Multiplication Sparsity Testbench 

This repository contains code for benchmarking matrix-matrix multiplication
on the Cerebras CS-2, release 1.8.0. This code was executed successfully on the 
Argonne Computing Leadership Facility's (ALCF) CS-2, but installations may differ on
other system. 

## Approach

This code seeks to measure runtime of matrix multiplication on Cerebras kernels 
by running many matrix multiplications to dominate any other noise/overhead and then
measuring overall runtime.

To use Cerebras kernels we must use one of their AI APIs. This code uses pytorch.

## Structure
Benchmarking code is located in `benchmarking/` and is based off of the modelzoos
FC MNIST code:
 - `model.py` contains the matrix multiplications being tested, phrased as a
    linear layer in a pytorch model. Multiplications are done between the weights
    of the linear layer and the matrix/vector formed by the input columns.
 - `data.py` is where the randomized "dataset" is generated. 
    Each sample becomes a column in an input matrix, so any tweaks to the structure
    of the inputs can be done here
 - `utils.py` contains, among other things, the function used to sparsify matrices
 - `run.py` contains the programs entry point and likely does not need to be touched
 - `prepare_data.py` is unused
 - The all-important `configs/` directory, which is used to specify the experiments.
    - `params.yaml` contains the default or template parameters for each run
    - `make_params.py` is a standalone script that reads `params.yaml` and generates
        a new params file for each trial. One execution is done per generated param file.
        It is by changing values between generated files that we control things
        like sparsity between runs.
    - `generated/` is the destination for each trial's param file. If not being
        overwritten by something of the same name, this directory must be emptied
        manually to avoid extra trials running.

## Installation
After cloning this repository, on the ALCF platform you can run `install.sh`. 
This has two main purposes:
 * Source the Cerebras Model Zoo. This code has run on the SlimPajama release
    (SHA 4e2a4e0) but later versions have come one
 * Locate and install the cerebras pytorch implementation

The script keeps track of the modelzoo and pytorch library using a virtual environment,
which must be activated every time you wish to run code by running `source setup.sh`.

## Usage

After sourcing `setup.sh`, a single trial based on `params.yaml` can be run with
`run_single.sh` or all trials can be generated and run with `run_benchmarks.sh`.
These files also control which output folder is used.

After running with sparsity, it's a good idea to double-check that weights
were correctly sparsified. Enter the path to the models `hdf5` file in `check_sparsity.py`
and see if the sparsity is correct. A common issue I've encountered is that too-large
matrices will be silently truncated, leaving 0's in the bottom chunk of the matrix: check
tail of the row sums to see that they aren't just zero. 

To read the runtimes performances, consult the `performance/performance.json` file
in the corresponding output directory.