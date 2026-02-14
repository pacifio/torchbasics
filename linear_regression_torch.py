import torch
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
w = torch.rand((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

learning_rate = 0.4
epochs = 20
for epoch in range(epochs):
	y_pred = X_train @ w+b
	loss = ((y_pred - y_train)**2).mean()
	loss.backward()
	with torch.no_grad():
		b -= learning_rate * b.grad #type:ignore
		w -= learning_rate * w.grad #type:ignore
		if b.grad is not None:
			b.grad.zero_()
		if w.grad is not None:
			w.grad.zero_()
		print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


X_new = X_test[:3]
with torch.no_grad():
	y_pred = X_new @ w+b

print(y_pred)
