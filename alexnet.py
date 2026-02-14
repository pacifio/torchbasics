import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import ImageFolder
from tqdm import tqdm

from dataset_loader import valid_dataset


class AlexNet(nn.Module):
	def __init__(self, classes=2, dropout_p=0.5) -> None:
		super().__init__()
		self.classes = classes
		self.feature_extractor = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(num_features=64),

			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(num_features=192),

			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=256),
		)

		self.avgpool = nn.AdaptiveAvgPool2d((6,6))
		self.head = nn.Sequential(
			nn.Dropout(dropout_p),
			nn.Linear(256*6*6, 4096),
			nn.ReLU(),
			nn.Dropout(dropout_p),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, classes)
		)

	def forward(self, x):
		batch_size = x.shape[0]
		x = self.feature_extractor(x)
		x = self.avgpool(x)
		x = x.reshape(batch_size, -1)
		x = self.head(x)
		return x


PATH_TO_DATA = "PetImages/"
normalizer = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) ### IMAGENET MEAN/STD ###
train_transforms = T.Compose([
	T.Resize((224, 224)),
	T.RandomHorizontalFlip(),
	T.ToTensor(),
	normalizer,
])

dataset = ImageFolder(PATH_TO_DATA, transform=train_transforms)
train_samples, test_samples = int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[train_samples, test_samples])

model = AlexNet()
epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

def train(model, device, epochs, optimizer, criterion, batch_size, trainloader, validloader):
		log_training = {"epoch": [],
                    "training_loss": [],
                    "training_acc": [],
                    "validation_loss": [],
                    "validation_acc": []}

		for epoch in range(1, 1+epochs):
			training_losses, training_accuracies = [],[]
			validation_losses, validation_accuracies = [], []

			model.train()
			for image, label in tqdm(trainloader):
				optimizer.zero_grad()
				out = model(image)
				loss = criterion(out, label)

				training_losses.append(loss.item())

				predictions = torch.argmax(out, dim=1)
				accuracy = (predictions==label).sum()/len(predictions)
				training_accuracies.append(accuracy.item())

				loss.backward()
				optimizer.step()

			model.eval()
			for image, label in tqdm(validloader):
				with torch.no_grad():
					out = model(image)
					loss = criterion(out, label)
					validation_losses.append(loss.item())

					predictions = torch.argmax(out, dim=1)
					accuracy = (predictions==label).sum()/len(predictions)
					validation_accuracies.append(accuracy.item())

			training_losses_mean, training_acc_mean = np.mean(training_losses), np.mean(training_accuracies)
			valid_loss_mean, valid_acc_mean = np.mean(validation_losses), np.mean(validation_accuracies)

			log_training["epoch"].append(epoch)
			log_training["training_loss"].append(training_losses_mean)
			log_training["training_acc"].append(training_acc_mean)
			log_training["validation_loss"].append(valid_loss_mean)
			log_training["validation_acc"].append(valid_acc_mean)
