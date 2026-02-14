import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data.dataloader import DataLoader

toTensor = T.Compose([
	T.ToImage(),
	T.ToDtype(torch.float32, scale=True)
])

train_and_valid_data = torchvision.datasets.FashionMNIST(
	root="datasets",
	train=True,
	download=True,
	transform=toTensor
)

test_data = torchvision.datasets.FashionMNIST(
	root="datasets",
	train=True,
	download=True,
	transform=toTensor
)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
	train_and_valid_data, [55_000, 5_000]
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

class ImageClassifier(nn.Module):
	def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes) -> None:
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Flatten(),
			nn.Linear(n_inputs, n_hidden1),
			nn.ReLU(),
			nn.Linear(n_hidden1, n_hidden2),
			nn.ReLU(),
			nn.Linear(n_hidden2, n_classes)
		)

	def forward(self, X):
		return self.mlp(X)

model = ImageClassifier(
	n_inputs=28*28,
	n_hidden1=300,
	n_hidden2=100,
	n_classes=10
)

# because of multiclass classification
xentropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

def train(model, optimizer, criterion, data_loader, epochs):
	model.train()
	for epoch in epochs:
		total_loss = 0.
		for X_batch, y_batch in data_loader:
			y_pred = model(X_batch)
			loss = criterion(y_pred, y_batch)
			loss.backward()
			total_loss += loss.item()
			optimizer.step()
			optimizer.zero_grad()
		mean_loss = total_loss/len(data_loader)
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.4f}")

train(model, optimizer, xentropy, train_loader, 20)
