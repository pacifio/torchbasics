import torch
import torch.nn as nn

layer = nn.Linear(40, 10)
layer.weight.data *= 6 ** 0.5
torch.zero_(layer.bias.data)

nn.init.kaiming_uniform(layer.weight)
nn.init.zeros_(layer.bias)

def use_he_init(module):
	if isinstance(module, nn.Linear):
		nn.init.kaiming_uniform(module.weight)
		nn.init.zeros_(module.bias)

model: nn.Module | nn.Sequential = nn.Sequential(
	nn.Linear(50, 40),
	nn.ReLU(),
	nn.Linear(40, 1),
	nn.ReLU()
)
model.apply(use_he_init)
