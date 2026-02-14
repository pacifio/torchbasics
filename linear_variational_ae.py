import random

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.auto import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

tensor_transforms = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]
)

train_set = MNIST("../../../data/mnist/", train=True, transform=tensor_transforms)
test_set = MNIST("../../../data/mnist/", train=False, transform=tensor_transforms)


class LinearVariationalAutoEncoder(nn.Module):
	def __init__(self, latent_dim=2) -> None:
		super().__init__()

		self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

		# mean of the gaussian distribution
		self.fn_mu = nn.Linear(32, latent_dim)
		self.fn_logvar = nn.Linear(32, latent_dim)

		self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )

	def forward_enc(self, x):
		x = self.encoder(x)

		mu = self.fn_mu(x)
		logvar = self.fn_logvar(x)
		sigma = torch.exp(0.5*logvar)
		noise = torch.randn_like(sigma, device=sigma.device)
		z = mu+sigma*noise

		return z, mu, logvar

	def forward_dec(self, x):
		return self.decoder(x)

	def forward(self, x):
		batch, channels, height, width = x.shape
		x = x.flatten(1)
		z, mu, logvar = self.forward_enc(x)
		dec = self.decoder(z)
		dec = dec.reshape(batch, channels, height, width)
		return z, dec, mu, logvar

def VAELoss(x, x_hat, mean, log_var, kl_weight=1, reconstruction_weight=1):
	pixel_mse = ((x-x_hat)**2)

	# flatten each image in batch to vector [b, C*H*W]
	pixel_mse = pixel_mse.flatten(1)
	reconstruction_loss = pixel_mse.sum(axis=-1).mean()
	kl = (1+log_var-mean**2-torch.exp(log_var)).flatten(1)
	kl_per_image = -0.5*torch.sum(kl, dim=-1)
	kl_loss = torch.mean(kl_per_image)

	return reconstruction_weight*reconstruction_loss + kl_weight*kl_loss

def train(model,
          kl_weight,
          train_set,
          test_set,
          batch_size,
          training_iterations,
          evaluation_iterations,
          model_type="VAE"):

    if model_type != "VAE": kl_weight = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_loss = []
    evaluation_loss = []

    encoded_data_per_eval = []
    train_losses = []
    evaluation_losses = []

    pbar = tqdm(range(training_iterations))

    train = True

    step_counter = 0
    while train:

        for images, labels in trainloader:

            images = images.to(device)

            if model_type == "VAE":
                encoded, decoded, mu, logvar = model(images)
                loss = VAELoss(images, decoded, mu, logvar, kl_weight) #type:ignore
            elif model_type == "AE":
                encoded, decoded = model(images)
                loss = torch.mean((images-decoded)**2)

            train_loss.append(loss.item()) #type:ignore

            loss.backward() #type:ignore
            optimizer.step()
            optimizer.zero_grad()

            if step_counter % evaluation_iterations == 0:

                model.eval()

                encoded_evaluations = []

                for images, labels in testloader:

                    images = images.to(device)

                    if model_type == "VAE":
                        encoded, decoded, mu, logvar = model(images)
                        loss = VAELoss(images, decoded, mu, logvar, kl_weight) #type:ignore
                    elif model_type == "AE":
                        encoded, decoded = model(images)
                        loss = torch.mean((images-decoded)**2)

                    evaluation_loss.append(loss.item()) #type:ignore

                    encoded, labels = encoded.cpu().flatten(1), labels.reshape(-1,1) #type:ignore

                    encoded_evaluations.append(torch.cat((encoded, labels), axis=-1)) #type:ignore


                encoded_data_per_eval.append(torch.concatenate(encoded_evaluations).detach())

                train_loss = np.mean(train_loss)
                evaluation_loss = np.mean(evaluation_loss)

                train_losses.append(train_loss)
                evaluation_losses.append(evaluation_loss)

                train_loss = []
                evaluation_loss = []

                model.train()

            step_counter += 1
            pbar.update(1)


            if step_counter >= training_iterations:
                print("Completed Training!")
                train = False
                break

    encoded_data_per_eval = [np.array(i) for i in encoded_data_per_eval]

    print("Final Training Loss", train_losses[-1])
    print("Final Evaluation Loss", evaluation_losses[-1])

    return model, train_losses, evaluation_losses, encoded_data_per_eval
