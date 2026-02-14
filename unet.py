import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import color
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

path_to_data = "../../data/carvana"

class CarvanaDataset(Dataset):
    """
    Carvana Class: This will do exactly what the ADE20K class was doing, but also include a random sampling
    of data as Carvana doesn't automatically split the dataset into training and validation (with an included seed)
    """
    def __init__(self, path_to_data, train=True, image_size=128, random_crop_ratio=(0.5, 1), seed=0, test_pct=0.1):
        self.path_to_data = path_to_data
        self.train = train
        self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio

        ### Get Path to Images and Segmentations ###
        self.path_to_images = os.path.join(self.path_to_data, "train")
        self.path_to_annotations = os.path.join(self.path_to_data, "train_masks")

        ### Get All Unique File Roots ###
        file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]

        ### Random Split Dataset into Train/Test ###
        random.seed(0)
        testing_data = random.sample(file_roots, int(test_pct*len(file_roots)))
        training_data = [sample for sample in file_roots if sample not in testing_data]
        random.seed(None)

        if self.train:
            self.file_roots = training_data
        else:
            self.file_roots = testing_data

        ### Store all Transforms we want ###
        self.resize = transforms.Resize(size=(self.image_size, self.image_size))
        self.normalize = transforms.Normalize(mean=(0.48897059, 0.46548275, 0.4294),
                                              std=(0.22861765, 0.22948039, 0.24054667))
        self.random_resize = transforms.RandomResizedCrop(size=(self.image_size, self.image_size))
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_roots)

    def __getitem__(self, idx):

        ### Grab File Root ###
        file_root = self.file_roots[idx]

        ### Get Paths to Image and Annotation ###
        image = os.path.join(self.path_to_images, f"{file_root}.jpg")
        annot = os.path.join(self.path_to_annotations, f"{file_root}_mask.gif")

        ### Load Image and Annotation ###
        image = Image.open(image).convert("RGB")
        annot = Image.open(annot)

        ### Train Image Transforms ###
        if self.train:

            ### Resize Image and Annotation ###
            if random.random() < 0.5:

                image = self.resize(image)
                annot = self.resize(annot)

            ### Random Resized Crop ###
            else:

                ### Get Smaller Side ###
                min_side = min(image.size)

                ### Get a Random Crop Size with Ratio ###
                random_ratio = random.uniform(self.min_ratio, self.max_ratio)

                ### Compute Crop Size ###
                crop_size = int(random_ratio * min_side)

                ### Get Parameters of Random Crop ###
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size)) #type:ignore

                ### Crop Image and Annotation ###
                image = TF.crop(image, i, j, h, w) #type:ignore
                annot = TF.crop(annot, i, j, h, w) #type:ignore

                ### Resize Image to Desired Image Size ###
                image = self.resize(image)
                annot = self.resize(annot)


            ### Random Horizontal Flip ###
            if random.random() < 0.5:
                image = self.horizontal_flip(image)
                annot = self.horizontal_flip(annot)

        ### Validation Image Transforms ###
        else:

            image = self.resize(image)
            annot = self.resize(annot)

        ### Convert Everything to Tensors ###
        image = self.totensor(image)
        annot = torch.tensor(np.array(annot), dtype=torch.float) # BCEWithLogits needs float tensor

        ### Normalize Image ###
        image = self.normalize(image)

        return image, annot

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, groupnorm_num_groups) -> None:
		super().__init__()

		self.groupnorm_1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

		self.groupnorm_2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")

		self.residiual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

	def forward(self, x):
		residual_connection = x

		x = self.groupnorm_1(1)
		x = F.relu(x)
		x = self.conv1(x)

		x = self.groupnorm_2(x)
		x = F.relu(x)
		x = self.conv2(x)

		x = x + self.residiual_connection(residual_connection)
		return x


class UpsampleBLock(nn.Module):
	def __init__(self, in_channels, out_channels, interpolate=False) -> None:
		super().__init__()

		if interpolate:
			self.upsample = nn.Sequential(
				nn.Upsample(
					scale_factor=2,
					mode="bilinear",
					align_corners=True
				),
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size=3,
					stride=1,
					padding="same"
				)
			)

		else:
			self.upsample = nn.ConvTranspose2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=2,
				stride=2,
			)


class UNET(nn.Module):
	def __init__(self,
		in_channels=3,
		num_classes=150,
		start_dim=64,
		dim_mults = (1,2,4,8),
		residual_blocks_per_group=1,
		groupnorm_num_group=16,
		interpolated_upsample=False,
	) -> None:
		super().__init__()

		self.input_image_channels = in_channels
		self.interpolate = interpolated_upsample

		channel_sizes = [start_dim*i for i in dim_mults]
		starting_channel_size, ending_channel_size = channel_sizes[0], channel_sizes[-1]

		self.encoder_config = []
		for idx, d in enumerate(channel_sizes):
			for _ in range(residual_blocks_per_group):
				self.encoder_config.append(((d, d), "residual"))
			self.encoder_config.append(((d, d), "downsample"))
			if idx < len(channel_sizes) - 1:
				self.encoder_config.append(((d, channel_sizes[idx+1]), "residual"))

		self.bottleneck_config = []
		for _ in range(residual_blocks_per_group):
			self.bottleneck_config.append(((ending_channel_size, ending_channel_size), "residual"))

		out_dim = ending_channel_size
		reversed_encoder_config = self.encoder_config[::-1]

		self.decoder_config = []
		for idx, (metadata, type) in enumerate(reversed_encoder_config):
			enc_in_channels, enc_out_channels = metadata
			concact_num_channels = out_dim + enc_out_channels
			self.decoder_config.append(((concact_num_channels, enc_in_channels), "residual"))

			if type == "downsample":
				self.decoder_config.append(((concact_num_channels, enc_in_channels), "upsample"))
			out_dim = enc_in_channels

		concact_num_channels = starting_channel_size*2
		self.decoder_config.append(((concact_num_channels, starting_channel_size), "residual"))

		self.conv_in_proj = nn.Conv2d(
			self.input_image_channels,
			starting_channel_size,
			kernel_size=3,
			padding="same"
		)

		self.encoder = nn.ModuleList()
		for metadata, type in self.encoder_config:
			if type == "residual":
				in_channels, out_channels = metadata
				self.encoder.append(
					ResidualBlock(
						in_channels=in_channels,
						out_channels=out_channels,
						groupnorm_num_groups=groupnorm_num_group
					)
				)

			elif type == "downsample":
				in_channels, out_channels = metadata
				self.encoder.append(
					nn.Conv2d(
						in_channels,
						out_channels,
						kernel_size=3,
						stride=2,
						padding=1
					)
				)

		self.bottleneck = nn.ModuleList()
		for (in_channels, out_channels), _ in self.bottleneck_config:
			self.bottleneck.append(
				ResidualBlock(
					in_channels=in_channels,
					out_channels=out_channels,
					groupnorm_num_groups=groupnorm_num_group
				)
			)

		self.decoder = nn.ModuleList()
		for metadata, type in self.decoder_config:
			if type == "residual":
				in_channels, out_channels = metadata
				self.decoder.append(
					ResidualBlock(
						in_channels=in_channels,
						out_channels=out_channels,
						groupnorm_num_groups=groupnorm_num_group
					)
				)

			elif type == "upsample":
				in_channels, out_channels = metadata
				self.decoder.append(
					UpsampleBLock(
						in_channels,
						out_channels,
						interpolate=self.interpolate
					)
				)

		self.conv_out_proj = nn.Conv2d(in_channels=starting_channel_size,out_channels=num_classes, kernel_size=3, padding="same")

	def forward(self, x):
		residuals = []
		x = self.conv_in_proj(x)
		residuals.append(x)

		for module in self.encoder:
			x = module(x)
			residuals.append(x)

		for module in self.bottleneck:
			x = module(x)

		for module in self.decoder:
			if isinstance(module, ResidualBlock):
				residual_tensor = residuals.pop()
				x = torch.cat([x, residual_tensor], dim=1)
				x = module(x)
			else:
				x = module(x)
		x = self.conv_out_proj(x)
		return x
