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

device = "cuda" if torch.cuda.is_available() else "cpu"


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=2):
        super().__init__()

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def forward(self, x, efficient=True):

        batch_size = x.shape[0]

        ### Bad Implementation That Requires Matrix Expansion ###
        if not efficient:

            # C: Codebook Size, L: Latent Dim

            ### Embedding: [C, L] -> [B, C, L]
            emb = self.embedding.weight.unsqueeze(0).repeat(batch_size,1,1)

            ### X: [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)

            ### [B, C]
            distances = torch.sum(((x - emb)**2), dim=-1)

        ### Alternative more Efficient Implementation ###
        else:
            ### Distance btwn every Latent and Code: (L-C)**2 = (L**2 - 2LC + C**2 ) ###

            ### L2: [B, L] -> [B, 1]
            L2 = torch.sum(x**2, dim=1, keepdim=True)

            ### C2: [C, L] -> [C]
            C2 = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)

            ### CL: [B,L]@[L,C] -> [B, C]
            CL = x@self.embedding.weight.t()

            ### [B, 1] - 2 * [B, C] + [C] -> [B, C]
            distances = L2 - 2*CL + C2

        ### Grab Closest Indexes, create matrix of corresponding vectors ###
        ### Closest: [B, 1]
        closest = torch.argmin(distances, dim=-1)

        ### Create Empty Quantized Latents Embedding ###
        # latents_idx: [B, C]
        quantized_latents_idx = torch.zeros(batch_size, self.codebook_size, device=x.device)

        ### Place a 1 at the Indexes for each sample for the codebook we want ###
        batch_idx = torch.arange(batch_size)
        quantized_latents_idx[batch_idx,closest] = 1

        ### Matrix Multiplication to Grab Indexed Latents from Embeddings ###

        # quantized_latents: [B, C] @ [C, L] -> [B, L]
        quantized_latents = quantized_latents_idx @ self.embedding.weight

        return quantized_latents


vq = VectorQuantizer(codebook_size=512,latent_dim=8)
rand = torch.randn(1024,8)
vq(rand)

class LinearVQVAE(nn.Module):
	def __init__(self, latent_dim, codebook_size) -> None:
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(32*32, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, latent_dim)
		)

		self.vq = VectorQuantizer(codebook_size, latent_dim)

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 32),
			nn.ReLU(),
			nn.Linear(32, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, 32*32),
		)

	def forward_enc(self, x):
		x = self.encoder(x)
		return x

	def quantize(self, z):
		codes = self.vq(z)
		codebook_loss = torch.mean((codes-z.detach())**2)
		commitment_loss = torch.mean((codes.detach() - z)**2)

		codes = z+(codes-z).detach()

		return codes, codebook_loss, commitment_loss

	def forward_dec(self, x):
		codes, codebook_loss, commitment_loss = self.quantize(x)
		decoded = self.decoder(codes)
		return codes, decoded, codebook_loss, commitment_loss

	def forward(self, x):
		batch, channels, height, width = x.shape
		x = x.flatten(1)
		latents = self.forward_enc(x)

		quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(latents)
		decoded = decoded.reshape(batch, channels, height, width)
		return latents, quantized_latents, decoded, codebook_loss, commitment_loss
