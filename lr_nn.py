import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

torch.manual_seed(42)
n_features = X_train.shape[1] # 8 input features
model = nn.Linear(in_features=n_features, out_features=1)

# print(model.bias)
# print(model.weight)

# print(model(X_train[:2]))

num_epochs = 20
learning_rate = 0.4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

def train(model: nn.Sequential | nn.Linear, optimizer, criterion, X_train, y_train, epochs):
	model.train()
	for epoch in range(epochs):
		y_pred = model(X_train)
		loss = criterion(y_pred, y_train)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

train(model, optimizer, mse, X_train, y_train, num_epochs)
