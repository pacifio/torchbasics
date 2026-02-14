## A very basic model

```python
n_features = X_train.shape[1]
w = torch.rand((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

learning_rate = 0.2
epochs = 20

for epoch in epochs:
	y_pred = X_train @ w + b
	loss = ((y_pred-y_train)**2).mean()
	loss.backward()
	with torch.no_grad():
		w -= learning_rate * w.grad
		b -= learning_Rate * b.grad
	w.grad.zero_()
	b.grad.zero_()
```

# Using nn.Module

```python
model = nn.Linear(n_features, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
	y_pred = model(X_train)
	loss = criterion(y_pred, y_train)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
````

# Using dataset

```python
from torch.data import TensorData, DataLoader

train_dataset = TensorData(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(epocs):
	total_loss = 0.
	for X_batch, y_batch in train_loader:
		y_pred = model(X_batch)
		loss = criterion(y_pred, y_batch)
		total_loss += loss.item()
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		...
	mean_loss = total_loss/len(train_loader)
	
```

# Using torchmetrics for eval

```python
rmse = torchmetrics.MeanSquaredError(squared=False)
def eval(model, metrics, data_loader):
	model.eval()
	metrics.reset()
	with torch.no_grad():
		for X_batch, y_batch in data_loader:
			y_pred = model(X_batch)
			metrics.update(y_pred, y_batch)
	return metrics.comptue()
```

# basic recap

```python
n_features = X_train.shape()[1]
w = torch.rand((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

epochs = 20
for epoch in range(epochs):
	y_pred = X_train @ w+b
	loss = ((y_pred - y_train)**2).mean()
	loss.backward()
	with torch.no_grad():
		w -= learning_rate * w.grad
		b -= learning_rate * b.grad
	w.grad.zero_()
	b.grad.zero_()
	
model = nn.Linear(n_features, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
	y_pred = model(X_train)
	loss = criterion(y_pred, y_train)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
```
	
## Explaining what these blocks are

### conv2d -> [batch, channels, W,H] -> [batch, channels, W', H']

conv2d basically extracts (given the parameters kernel size and stride)
and creates new output channels with the updated width and height (which is derived from the calculated blocks from kernel and stride)

### batchnorm2d
it creates normalised distributions along with learned weights for consistency
this results in so let's say we might have vanishing gradients over multiple blocks of computes 
that's causing the gradient to steer towards 0 and activation functions are not helping and 
this normalisation technique can help maintain a certain consistency between dramatic data change 
via mean and variance computation and also i just learned about the fact that batchnorms have learned weights that are internal

so in contrast to all of this
> conv2d is working per channel and updating the weights for w' and h' and creating a new out channel
which works with the kernel size and the stride

> what batchnorm does
it works across channels and operates on B/W/H
and per channel it only updates the data inside the channel and
computes the mean and the variance and also has learnable weights via
scale and shift for backpropagation via gradient tracking

## Pooling
so we can downsample or basically zoom out from each tensor (images here)
it's mostly a design choice for optimisation
for example if we pass tensors in to pool objects it will create
[batch_size, channels, W, H] -> [batch_size, channels, W', H']

## Resnets
During training (backpropagation), the skip connection acts as a "highway". 
The gradient signal can flow through the skip connection without being multiplied 
by the weights of every single layer, preventing it from disappearing.

## LSTM
essentially what they are, are networks with gates that convey a set of long term memory via cells
and short term (recent) memories via hidden states, and thanks to BPTT (back propagation through time)
the gradients avoid vanishing because the networks know what information to forget via forget gates
and what information to convey via input/output gates based on the loss function

## again reacp

```python
model = nn.Linear()
optimizer = torch.optim.utils.AdamW(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
for epoch in range(epochs):
	model.train()
	for x_batch, y_batch in train_loader:
		optimizer.no_grad()
		y_pred = model(x_batch)
		loss = criterion(y_pred, y_batch)
		total_loss += loss.item()
		loss.backward()
		optimizer.step()
```

## now in order to learn about stable diff and generate gans we must know about

> autoencoders
	> residual autoencoders
	> variational autoencoders
	> image segmentation via UNETs
	> generative adversarial networks or GANs
	> diffusion
	> latent diff
	
## autoencoders what they are?

so essentially what's happening is that the model is learning encoding and decoding
by itself because the encoder works to reduce the dimension 
and the decoder is learning to upscale the dimension 
from encoder so the training data essentially is just itself.

so these are learned compression sort of.
