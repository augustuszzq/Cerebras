import torch
import torch.nn.functional
import cerebras_pytorch as cbtorch
from functools import reduce

cp_path = "outputs_nan/model_0.33/initial_state_1689950135.127909.hdf5"

state_dict = cbtorch.load(cp_path)

# print(state_dict)
print(state_dict["model"].keys())
weights = state_dict["model"]["mat_mul_9.lin_layer.weight"]

print(weights)
print(torch.sum(weights, axis = 1))
print(weights.shape)


n_total = reduce(lambda x, y: x * y, weights.shape)
n_nonzero = torch.count_nonzero(weights).item()

density = n_nonzero / n_total
sparsity = 1 - density
print(f"Sparsity: {sparsity}")
