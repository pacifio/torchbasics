import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

warnings.filterwarnings("ignore")

# the size of the image after a convolution is applied
# (width-kernelSize+2 * padding)/stride + 1

class MyConv2d(nn.Module):
	def __init__(self,in_channels, out_channels, kernel_size, stride, padding) -> None:
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.linear = nn.Linear(
			in_channels*kernel_size*kernel_size, out_channels,
			bias=True
		)

	def forward(self, x: torch.Tensor):
		batch_size, channels, height, width = x.size()
		assert channels == self.in_channels

		# unfold the input into patches
		# shape: [batch_size, in_channels*kernel_size*kernel_size, num_patches]
		patches = nn.functional.unfold(
			x,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding
		)

		_, num_kernel_coeffiicents, num_patches =  patches.shape

		# batch_size*num_patches, in_channel*kernel_size*kernel_size
		patches = patches.transpose(1, 2).reshape(-1, num_kernel_coeffiicents)
		conv_output = self.linear(patches)
		conv_output = conv_output.view(batch_size, self.out_channels, -1).transpose(1, 2)

		output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
		output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
		output = conv_output.transpose(1, 2).view(batch_size, self.out_channels, output_height, output_width)

		return output

myconv = MyConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)
torchconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0)

### Create a random tensor
rand = torch.randn(4,3,128,128)

myconv_out = myconv(rand)
torchconv_out = torchconv(rand)

print("Output of My Convolution:",  myconv_out.shape)
print("Output of PyTorch Convolution:", torchconv_out.shape)
