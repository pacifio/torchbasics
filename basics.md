# PyTorch From Zero to Building Any Model

I'll teach you everything systematically, from tensors to building complex architectures.

## 1. Tensors: The Foundation

Tensors are PyTorch's core data structure - think of them as multi-dimensional arrays with GPU acceleration.

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])                    # From list
x = torch.zeros(3, 4)                           # 3x4 tensor of zeros
x = torch.ones(2, 3, 4)                         # 2x3x4 tensor of ones
x = torch.randn(2, 3)                           # Random normal distribution
x = torch.rand(2, 3)                            # Random uniform [0, 1)
x = torch.arange(0, 10, 2)                      # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)                     # [0.0, 0.25, 0.5, 0.75, 1.0]

# From numpy
import numpy as np
arr = np.array([1, 2, 3])
x = torch.from_numpy(arr)
back_to_numpy = x.numpy()

# Tensor properties
print(x.shape)        # Size of each dimension
print(x.dtype)        # Data type (float32, int64, etc.)
print(x.device)       # cpu or cuda
```

### Tensor Operations

```python
# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

c = a + b              # Element-wise addition
c = a * b              # Element-wise multiplication
c = a @ b              # Dot product (for 1D)
c = torch.matmul(a, b) # Matrix multiplication

# Reshaping
x = torch.randn(4, 4)
y = x.view(16)         # Reshape to 1D (must have same # of elements)
z = x.view(-1, 8)      # -1 means "infer this dimension"
w = x.reshape(2, 8)    # Similar to view but copies if needed

# Indexing and slicing
x = torch.randn(3, 4)
print(x[0])            # First row
print(x[:, 1])         # Second column
print(x[1, 2])         # Element at (1, 2)

# Concatenation and stacking
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.cat([a, b], dim=0)      # Concatenate along dimension 0 -> (4, 3)
d = torch.stack([a, b], dim=0)    # Stack creates new dimension -> (2, 2, 3)
```

### GPU Operations

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Move tensor to GPU
x = torch.randn(3, 3)
x = x.to(device)

# Or create directly on GPU
x = torch.randn(3, 3, device=device)

# Move back to CPU
x = x.cpu()
```

## 2. Autograd: Automatic Differentiation

This is PyTorch's magic - it automatically computes gradients for backpropagation.

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Forward pass
z = x * y + y**2

# Backward pass - compute gradients
z.backward()

# Access gradients
print(x.grad)  # dz/dx = y = 3.0
print(y.grad)  # dz/dy = x + 2y = 2.0 + 6.0 = 8.0

# Multiple operations
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.mean()  # (2 + 4 + 6) / 3 = 4.0

z.backward()
print(x.grad)  # [2/3, 2/3, 2/3]
```

### Gradient Management

```python
# Zero gradients (important in training loops!)
x.grad.zero_()

# Detach from computation graph
y = x.detach()  # y doesn't track gradients

# Temporarily disable gradient tracking
with torch.no_grad():
    y = x * 2  # No gradients computed here
```

## 3. Neural Network Basics with `nn.Module`

All models inherit from `nn.Module`. This is the standard pattern:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()  # Always call parent constructor
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)

# Forward pass
x = torch.randn(5, 10)  # Batch of 5 samples
output = model(x)        # Calls forward() automatically
print(output.shape)      # torch.Size([5, 2])
```

## 4. Common Layers and Modules

### Linear (Fully Connected) Layers

```python
# Linear layer: y = xW^T + b
fc = nn.Linear(in_features=10, out_features=20)
x = torch.randn(5, 10)
output = fc(x)  # Shape: (5, 20)
```

### Convolutional Layers

```python
# 2D Convolution for images
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=16,    # 16 filters
    kernel_size=3,      # 3x3 kernel
    stride=1,
    padding=1
)

x = torch.randn(8, 3, 32, 32)  # Batch of 8 RGB 32x32 images
output = conv(x)                # Shape: (8, 16, 32, 32)

# 1D Convolution for sequences
conv1d = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3)
```

### Pooling Layers

```python
# Max pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 16, 32, 32)
output = maxpool(x)  # Shape: (8, 16, 16, 16) - halved spatial dims

# Average pooling
avgpool = nn.AvgPool2d(kernel_size=2)

# Adaptive pooling (output size specified)
adaptive = nn.AdaptiveAvgPool2d((1, 1))  # Always outputs 1x1
```

### Activation Functions

```python
# ReLU
x = torch.randn(5, 10)
relu = nn.ReLU()
output = relu(x)
# Or functional form
output = F.relu(x)

# Other common activations
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
gelu = nn.GELU()  # Used in transformers
softmax = nn.Softmax(dim=1)  # For classification
```

### Normalization Layers

```python
# Batch Normalization
bn = nn.BatchNorm2d(num_features=16)
x = torch.randn(8, 16, 32, 32)
output = bn(x)

# Layer Normalization (used in transformers)
ln = nn.LayerNorm(normalized_shape=512)

# Dropout for regularization
dropout = nn.Dropout(p=0.5)
```

### Recurrent Layers

```python
# LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

x = torch.randn(8, 50, 10)  # (batch, seq_len, input_size)
output, (hn, cn) = lstm(x)   # output: (8, 50, 20)

# GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
```

### Attention and Transformers

```python
# Multi-head attention
attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    batch_first=True
)

# Transformer encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048
)
```

## 5. Building Complete Models

### Pattern 1: Sequential Model (Simple)

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)
```

### Pattern 2: Custom Module (Flexible)

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch, 3, 32, 32)
        
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 128, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*4*4)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = CNN(num_classes=10)
```

### Pattern 3: Nested Modules

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # Skip connection
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Stack multiple residual blocks
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## 6. Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()  # Most common for multi-class
outputs = model(x)
loss = criterion(outputs, labels)

# Binary classification
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy

# Regression
criterion = nn.MSELoss()           # Mean squared error
criterion = nn.L1Loss()            # Mean absolute error

# Custom loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        loss = ((predictions - targets) ** 2).mean()
        return loss

criterion = CustomLoss()
```

## 7. Optimizers

```python
import torch.optim as optim

# Get model parameters
model = SimpleNet(10, 20, 2)

# Common optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Different learning rates for different layers
optimizer = optim.Adam([
    {'params': model.conv_layers.parameters(), 'lr': 0.001},
    {'params': model.fc_layers.parameters(), 'lr': 0.01}
])
```

## 8. Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. Define model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Data loaders (assume these are defined)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 4. Training loop
num_epochs = 10

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set model to training mode
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    # Calculate epoch statistics
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # No gradients needed for validation
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
```

## 9. Model Utilities

### Saving and Loading Models

```python
# Save entire model
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# Save only state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))

# Save checkpoint with optimizer state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### Model Information

```python
# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {count_parameters(model)}')

# Print model architecture
print(model)

# Access specific layers
for name, param in model.named_parameters():
    print(f'{name}: {param.shape}')

# Freeze layers
for param in model.conv_layers.parameters():
    param.requires_grad = False
```

### Inference

```python
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    probabilities = F.softmax(output, dim=1)
    predicted_class = output.argmax(1)
    print(f'Predicted class: {predicted_class.item()}')
```

## 10. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Usage in training loop
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    
    scheduler.step()  # For StepLR, CosineAnnealing
    # scheduler.step(val_loss)  # For ReduceLROnPlateau
```

## 11. Common Model Architectures Quick Reference

### CNN for Image Classification

```python
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### RNN for Sequence Processing

```python
class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        x = self.fc(hidden[-1])
        return x
```

### Autoencoder

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

## Key Takeaways

1. **Tensors** are your data containers (like NumPy arrays but with GPU support)
2. **nn.Module** is the base class for all models - implement `__init__` and `forward`
3. **Autograd** handles backpropagation automatically - just call `loss.backward()`
4. **Training loop pattern**: zero_grad → forward → loss → backward → step
5. **Always move data and model to the same device** (CPU or GPU)
6. **model.train()** vs **model.eval()** affects dropout and batch norm behavior
7. **torch.no_grad()** disables gradient tracking for inference

That's everything you need to start building models in PyTorch! The pattern is always the same: define your model architecture, create a loss function and optimizer, then train with the standard loop. Start simple and gradually add complexity as needed.
