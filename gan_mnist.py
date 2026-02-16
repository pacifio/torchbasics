import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm


class MNISTGenerator(nn.Module):
	def __init__(self, latent_dimension) -> None:
		super().__init__()

		self.generator = nn.Sequential(
			nn.Linear(latent_dimension, 256),
			nn.LeakyReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(256, 512),
			nn.LeakyReLU(),
			nn.Dropout(p=0.2),
			nn.Linear(512, 1024),
			nn.Dropout(0.2),
			nn.Linear(1024, 784),
			nn.Tanh()
		)

	def forward(self, noise):
		batch_size = noise.shape[0]
		generated = self.generator(noise)
		return generated.reshape(batch_size, 1, 28, 28)

class MNISTDiscriminator(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.discriminator = nn.Sequential(
			nn.Linear(784, 1024),
			nn.LeakyReLU(),
			nn.Dropout(0.2),
			nn.Linear(1024, 512),
			nn.LeakyReLU(),
			nn.Dropout(0.2),
			nn.Linear(512, 256),
			nn.LeakyReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 1),
		)

	def forward(self, x):
		batch_size = x.shape[0]
		x = x.reshape(batch_size, -1)
		return self.discriminator(x)

latent_dimension = 100
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 200

### Define Datasets ###
tensor2image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = MNIST("../../../data", transform=tensor2image_transforms)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

def train_unconditional_gan(
	generator,
	discriminator,
	generator_optimizer,
	discriminator_optimizer,
	dataloader,
	label_smoothing=0.05,
	epochs=200,
	device="cpu",
	plot_generation_feq=20,
	num_gens=10,
):
	loss_func = nn.BCEWithLogitsLoss()
	gen_losses, disc_losses = [], []

	for epoch in tqdm(range(epochs)):
		generator_epoch_losses = []
		discriminator_epoch_losses = []

		for images, _ in dataloader:
			batch_size = images.shape[0]
			images = images.to(device)

			## train discriminator

			noise = torch.randn(batch_size, latent_dimension, device=device)
			generated_labels = torch.zeros(batch_size, 1, device=device) + label_smoothing
			true_labels = torch.ones(batch_size, 1, device=device) - label_smoothing
			generated_images = generator(noise).detach()

			real_discriminator_pred = discriminator(images)
			gen_discriminator_pred = discriminator(generated_images)

			real_loss = loss_func(real_discriminator_pred, true_labels)
			fake_loss = loss_func(gen_discriminator_pred, generated_labels)

			discriminator_loss = (real_loss+fake_loss)/2
			discriminator_epoch_losses.append(discriminator_loss.item())

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()

			## train generator
			noise = torch.randn(batch_size, latent_dimension, device=device)
			generated_images = generator(noise)
			gen_discriminator_pred = discriminator(generated_images)

			generator_loss = loss_func(gen_discriminator_pred, true_labels)
			generator_epoch_losses.append(generator_loss.item())

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

		generator_epoch_losses = np.mean(generator_epoch_losses) #type:ignore
		discriminator_epoch_losses = np.mean(discriminator_epoch_losses) #type:ignore

		if epoch % plot_generation_feq == 0:
			print(f"epoch: {epoch}/{epochs} | generator loss : {generator_epoch_losses} | discriminator loss : {discriminator_epoch_losses}")

		gen_losses.append(generator_epoch_losses)
		disc_losses.append(discriminator_epoch_losses)

		if epoch % plot_generation_feq == 0:
			generator.eval()
			with torch.no_grad():
				noise_sample = torch.randn(num_gens, latent_dimension, device=device)
				generated_imgs = generator(noise_sample).to("cpu")

				fig, ax = plt.subplots(1, num_gens, figsize=(15,5))
				for i in range(num_gens):
					img = (generated_imgs[i].squeeze()+1)/2
					ax[i].imshow(img.numpy(), cmap="gray")
					ax[i].set_axis_off()
				plt.show()

			generator.train()

	return generator, discriminator, gen_losses, disc_losses
