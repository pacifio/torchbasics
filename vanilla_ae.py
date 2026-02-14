import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

### GENERATE ANIMATIONS ###
generate_anim = False

# I am resizing the 28x28 image to 32x32 just so its a power of 2!
transform = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
)

train_set = MNIST("../../../data/mnist/", train=True, transform=transform)
test_set = MNIST("../../../data/mnist/", train=False, transform=transform)

### SET DEVICE ###
device = "cuda" if torch.cuda.is_available() else "cpu"


class VanillaAutoEncoder(nn.Module):
	def __init__(self, bottleneck_size=2) -> None:
		super().__init__()

		self.encoder = nn.Sequential(
			nn.Linear(32*32, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, bottleneck_size)
		)

		self.decoder = nn.Sequential(
			nn.Linear(bottleneck_size, 32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, 32*32),
		)

	def forward_enc(self, x):
		return self.encoder(x)

	def forward_dec(self, x):
		x = self.decoder(x)
		x = x.reshape(-1, 1, 32, 32)
		return x

	def forward(self, x):
		batch, channels, height, width = x.shape
		x = x.flatten(1)
		enc = self.forward_enc(x)
		dec = self.forward_dec(enc)
		return enc, dec

def train(model,
	train_set,
	test_set,
	batch_size,
	training_iterations,
	evaluation_iterations,
	verbose=False):

	print("Training model")
	print(model)
	device = "cpu"
	model = model.to(device)

	trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	testloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

	train_loss = []
	evaluation_loss = []
	train_losses = []
	evaluation_losses = []

	encoded_data_per_level = []

	pbar = tqdm(range(training_iterations))

	train = True
	step_counter = 0
	while train:
		for images, labels in trainloader:
			images = images.to(device)
			encoded, reconstruction = model(images)

			loss = torch.mean((images-reconstruction)**2)
			train_loss.append(loss.item()) #type:ignore

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			if step_counter % evaluation_iterations == 0:
				model.eval()
				encoded_evaluations = []

				for images, labels in testloader:
					images = images.to(device)
					encoded, reconstruction = model(images)
					evaluation_loss.append(loss.item()) #type:ignore

					encoded, labels = encoded.cpu().flatten(1), labels.reshape(-1, 1)
					encoded_evaluations.append(torch.cat((encoded, labels), dim=-1))

				encoded_data_per_level.append(torch.concatenate(
					encoded_evaluations
				).detach())

				train_loss = np.mean(train_loss)
				evaluation_loss = np.mean(evaluation_loss)
				train_losses.append(train_loss)
				evaluation_losses.append(evaluation_loss)

				if verbose:
					print("training loss", train_loss)
					print("evaluation loss", evaluation_loss)

				train_loss = []
				evaluation_loss = []
				model.train()

			step_counter += 1
			pbar.update(1)

			if step_counter >= training_iterations:
				print("completed training!")
				train = False
				break

	encoded_data_per_eval = [np.array(i) for i in encoded_data_per_level]

	print("Final Training Loss", train_losses[-1])
	print("Final Evaluation Loss", evaluation_losses[-1])

	return model, train_losses, evaluation_losses, encoded_data_per_eval


vanilla_model = VanillaAutoEncoder(bottleneck_size=2)
vanilla_model, train_losses, evaluation_losses, vanilla_encoded_data = train(vanilla_model,
                                                                             train_set,
                                                                             test_set,
                                                                             64,
                                                                             25000,
                                                                             250)
