import torch
import torch.nn as nn
import torchmetrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = "cpu"

housing = fetch_california_housing(download_if_missing=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42 #type:ignore
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)
means = X_train.mean(dim=0, keepdim=True)
stds = X_train.std(dim=0, keepdim=True)
X_train = (X_train-means)/stds
X_valid = (X_valid-means)/stds
X_test = (X_test-means)/stds

y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

n_features = X_train.shape[1]
torch.manual_seed(42)
model = nn.Sequential(
	nn.Linear(n_features, 50),
	nn.ReLU(),
	nn.Linear(50, 40),
	nn.ReLU(),
	nn.Linear(40, 1),
)

learning_rate = 0.1
epochs = 20
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

def train(model, optimizer, criterion, train_loader, ecpohs):
	model.train()
	for epoch in range(epochs):
		total_loss = 0.
		for X_batch, y_batch in train_loader:
			y_pred = model(X_batch)
			loss = criterion(y_pred, y_batch)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		mean_loss = total_loss / len(train_loader)
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.4f}")

train(model, optimizer, mse, train_loader, epochs)

def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
	model.eval()
	metrics = []
	with torch.no_grad():
		for X_batch, y_batch in data_loader:
			y_pred = model(X_batch)
			metric = metric_fn(y_pred, y_batch)
			metrics.append(metric)
	return aggregate_fn(torch.stack(metrics))

valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32)
valid_mse = evaluate(model, valid_loader, mse)
print(valid_mse)

def rmse_mine(y_pred, y_true):return ((y_pred - y_true) ** 2).mean().sqrt()

rmse = torchmetrics.MeanSquaredError(squared=False)

def eval_tm(model: torch.nn.Sequential, data_loader, metric: torchmetrics.MeanSquaredError):
	model.eval()
	metric.reset()
	with torch.no_grad():
		for X_batch, y_batch in data_loader:
			y_pred = model(X_batch)
			metric.update(y_pred, y_batch)
	return metric.compute()

class WideAndDeepDataset(torch.utils.data.Dataset):
	def __init__(self, X_wide, X_deep, y) -> None:
		super().__init__()
		self.X_wide = X_wide
		self.X_deep = X_deep
		self.y = y

	def __len__(self) -> int:
		return len(self.y)

	def __getitem__(self, index):
		input_dict = {
			"X_wide": self.X_wide[index],
			"X_deep": self.X_deep[index]
		}
		return input_dict, self.y[index]

train_data_named = WideAndDeepDataset(
	X_wide=X_train[:, :5],
	X_deep=X_train[:, :2],
	y=y_train
)
train_loader_named = DataLoader(train_data_named, batch_size=32, shuffle=True)
