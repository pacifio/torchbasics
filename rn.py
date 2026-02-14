import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
	def __init__(self, in_planes, planes, downsample=None, middle_conv_stride=1, residual=True) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=middle_conv_stride)
		self.bn2 = nn.BatchNorm2d(planes)

		self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1)
		self.bn3 = nn.BatchNorm1d(planes*4)
		self.relu = nn.ReLU()

		self.downsample = downsample
		self.residual = residual

	def forward(self, x):
		identity = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.conv3(x)
		x = self.bn3(x)

		if self.residual:
			if self.downsample is not None:
				identity = self.downsample(identity)
			x = x + identity
		x = self.relu(x)
		return x

class ResNet(nn.Module):
	def __init__(self, layer_counts, num_channels=3, num_classes=2, residual=True) -> None:
		super().__init__()
		self.residual = residual
		self.inplanes = 64 # starting number of planes to map from input channels

		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm1d(self.inplanes)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layers(layer_counts[0], planes=64, stride=1)
		self.layer2 = self._make_layers(layer_counts[1], planes=128, stride=2)
		self.layer3 = self._make_layers(layer_counts[2], planes=256, stride=2)
		self.layer4 = self._make_layers(layer_counts[3], planes=512, stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.fc = nn.Linear(512*4, num_classes)

	def _make_layers(self, num_residual_blocks, planes, stride):
		downsample = None
		layers = nn.ModuleList() # create a module list to store all our convolutions
		if stride != 1 or self.inplanes != planes*4:
			downsample = nn.Sequential(
				nn.Conv2d(
					self.inplanes,
					planes*4,
					kernel_size=1,
					stride=stride
				),
				nn.BatchNorm2d(
					planes*4
				)
			)
		layers.append(ResidualBlock(
			in_planes=self.inplanes,
			planes=planes,
			downsample=downsample,
			middle_conv_stride=stride,
			residual=self.residual
		))

		self.inplanes = 4

		for _ in range(num_residual_blocks-1):
			layers.append(
				ResidualBlock(
					in_planes=self.inplanes,
					planes=planes,
					residual=self.residual
				)
			)

		return nn.Sequential(*layers)
