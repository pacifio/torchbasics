# Learning PyTorch by Building a SuperTuxKart AI
## A Practical, Project-Driven Guide to Deep Learning

**Author's Note:** Forget those 600-page textbooks filled with equations you'll never use. This guide teaches you PyTorch by building something real: an AI that learns to play SuperTuxKart by watching you play. You'll learn tensors, neural networks, training loops, and everything in between—all while building a complete project from scratch.

---

## Table of Contents

1. [Introduction: Why This Project?](#chapter-1-introduction)
2. [PyTorch Fundamentals: Tensors and Computation](#chapter-2-pytorch-fundamentals)
3. [Screen Capture: Getting Game Data](#chapter-3-screen-capture)
4. [Neural Networks: The Brain of Your AI](#chapter-4-neural-networks)
5. [Data Loading: Feeding Your Model](#chapter-5-data-loading)
6. [Training: Teaching Your AI](#chapter-6-training)
7. [Inference: Letting Your AI Play](#chapter-7-inference)
8. [Advanced Topics: Making It Better](#chapter-8-advanced-topics)
9. [Debugging and Optimization](#chapter-9-debugging)
10. [Next Steps: Where to Go From Here](#chapter-10-next-steps)

---

# Chapter 1: Introduction

## Why This Project?

You want to learn PyTorch. You could read a textbook, work through MNIST digit classification for the 100th time, or... you could build an AI that plays a racing game. Which sounds more fun?

This project teaches you:
- **Tensors**: The fundamental data structure
- **Neural Networks**: CNNs for vision, fully connected layers for decisions
- **Training Loops**: The heart of machine learning
- **Real-time Inference**: Making your AI actually do something
- **Data Pipelines**: How to feed data efficiently
- **GPU Acceleration**: Making it fast

## What We're Building

```
Your Gameplay → Screen Capture → Dataset → Neural Network → AI Player
                    ↓                           ↓
                Save frames              Predict actions
                Save inputs              Control game
```

**Phase 1:** Record your gameplay (frames + keyboard inputs)  
**Phase 2:** Build a neural network that learns from your data  
**Phase 3:** Train the network  
**Phase 4:** Let it play autonomously  

## Prerequisites

```bash
pip install torch torchvision numpy opencv-python pyobjc-framework-Quartz pyobjc-framework-Cocoa pillow
```

**What you should know:**
- Basic Python (functions, classes, loops)
- Basic understanding of what neural networks are (we'll explain the rest)
- How to run Python scripts

**What you don't need to know:**
- Advanced math (we'll explain what's necessary)
- Deep learning theory (you'll learn by doing)
- PyTorch (that's why you're here!)

---

# Chapter 2: PyTorch Fundamentals

## Understanding Tensors

Before we write any code, you need to understand the fundamental building block of PyTorch: the **tensor**.

### What is a Tensor?

Think of a tensor as a multi-dimensional array that can live on your GPU and knows how to compute gradients.

```python
import torch

# A scalar (0D tensor)
scalar = torch.tensor(3.14)

# A vector (1D tensor)
vector = torch.tensor([1, 2, 3, 4])

# A matrix (2D tensor)
matrix = torch.tensor([[1, 2], [3, 4]])

# A 3D tensor (like an RGB image)
image = torch.zeros(3, 224, 224)  # 3 channels, 224x224 pixels

# A 4D tensor (like a batch of images)
batch = torch.zeros(32, 3, 224, 224)  # 32 images, 3 channels, 224x224
```

**Key Insight:** In our project, game frames will be 4D tensors: `(batch_size, channels, height, width)`

### Tensor Operations

```python
# Creating tensors
x = torch.randn(3, 4)  # Random 3x4 tensor
y = torch.zeros(3, 4)   # All zeros
z = torch.ones(3, 4)    # All ones

# Basic operations
a = x + y               # Element-wise addition
b = x * 2               # Scalar multiplication
c = torch.matmul(x, x.T)  # Matrix multiplication

# Shape operations
x_flat = x.view(-1)     # Flatten to 1D
x_reshaped = x.view(2, 6)  # Reshape to 2x6

print(f"Shape: {x.shape}")  # torch.Size([3, 4])
print(f"Device: {x.device}")  # cpu or cuda
```

### GPU Acceleration

```python
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # For Apple Silicon
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Move tensor to device
x = torch.randn(1000, 1000)
x_gpu = x.to(device)

# All operations now happen on GPU
result = torch.matmul(x_gpu, x_gpu)
```

**For our project:** We'll use MPS (Metal Performance Shaders) on your Mac for GPU acceleration.

### Automatic Differentiation (Autograd)

This is PyTorch's superpower: automatic gradient computation.

```python
# Create a tensor that requires gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Define a computation
y = x ** 2 + 3 * x + 1

# Compute gradients
y.backward()

# Access the gradient dy/dx
print(x.grad)  # tensor([7.]) because dy/dx = 2x + 3 = 2(2) + 3 = 7
```

**Why this matters:** During training, we compute how wrong our predictions are (loss), then use `.backward()` to compute gradients that tell us how to adjust our neural network's parameters.

---

# Chapter 3: Screen Capture

## The Data Collection System

Before we can train an AI, we need data. Lots of it. We'll build a system that:
1. Finds the SuperTuxKart window
2. Captures frames at 30 FPS
3. Records your keyboard inputs
4. Saves everything to disk

### Architecture Overview

```python
import cv2
import numpy as np
import torch
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
    CGWindowListCreateImage,
    CGRectNull,
    kCGWindowImageDefault
)
from Cocoa import NSURL
import time
from datetime import datetime
import json
from pynput import keyboard
import threading
```

### Finding the Game Window

```python
def find_window_by_name(window_name):
    """
    Find a window by its title using Quartz.
    Returns window bounds if found, None otherwise.
    """
    window_list = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID
    )
    
    for window in window_list:
        if window_name.lower() in window.get('kCGWindowName', '').lower():
            bounds = window['kCGWindowBounds']
            return {
                'x': int(bounds['X']),
                'y': int(bounds['Y']),
                'width': int(bounds['Width']),
                'height': int(bounds['Height']),
                'window_id': window['kCGWindowNumber']
            }
    return None
```

### Capturing Frames

```python
def capture_window_region(x, y, width, height):
    """
    Capture a specific region of the screen.
    Returns numpy array in RGB format.
    """
    from Quartz import (
        CGWindowListCreateImage,
        CGRectMake,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowImageDefault
    )
    from CoreGraphics import CGRectMake
    
    # Define region
    region = CGRectMake(x, y, width, height)
    
    # Capture
    image = CGWindowListCreateImage(
        region,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
        kCGWindowImageDefault
    )
    
    if image is None:
        return None
    
    # Convert to numpy array
    width = CGImageGetWidth(image)
    height = CGImageGetHeight(image)
    bytesperrow = CGImageGetBytesPerRow(image)
    pixeldata = CGDataProviderCopyData(CGImageGetDataProvider(image))
    
    # Convert to numpy array
    img_array = np.frombuffer(pixeldata, dtype=np.uint8)
    img_array = img_array.reshape((height, bytesperrow // 4, 4))
    img_array = img_array[:, :width, :3]  # Remove alpha channel
    
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
```

### Recording Keyboard Inputs

```python
class InputRecorder:
    """
    Records keyboard inputs with timestamps.
    Maps WASD and arrow keys to game actions.
    """
    def __init__(self):
        self.current_keys = set()
        self.input_history = []
        self.recording = False
        self.lock = threading.Lock()
        
        # Define key mappings
        self.action_keys = {
            'w': 'accelerate',
            'a': 'left',
            's': 'brake',
            'd': 'right',
            'up': 'accelerate',
            'left': 'left',
            'down': 'brake',
            'right': 'right',
            'space': 'drift'
        }
    
    def on_press(self, key):
        if not self.recording:
            return
            
        try:
            key_char = key.char.lower() if hasattr(key, 'char') else key.name
            if key_char in self.action_keys:
                with self.lock:
                    self.current_keys.add(key_char)
        except AttributeError:
            pass
    
    def on_release(self, key):
        if not self.recording:
            return
            
        try:
            key_char = key.char.lower() if hasattr(key, 'char') else key.name
            if key_char in self.action_keys:
                with self.lock:
                    self.current_keys.discard(key_char)
        except AttributeError:
            pass
    
    def get_current_action_vector(self):
        """
        Convert current key presses to action vector.
        Returns: [accelerate, brake, left, right, drift]
        """
        with self.lock:
            keys = self.current_keys.copy()
        
        action = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Accelerate
        if 'w' in keys or 'up' in keys:
            action[0] = 1.0
        
        # Brake
        if 's' in keys or 'down' in keys:
            action[1] = 1.0
        
        # Left
        if 'a' in keys or 'left' in keys:
            action[2] = 1.0
        
        # Right
        if 'd' in keys or 'right' in keys:
            action[3] = 1.0
        
        # Drift
        if 'space' in keys:
            action[4] = 1.0
        
        return action
    
    def start_recording(self):
        self.recording = True
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        listener.start()
        return listener
```

### The Data Collector

```python
class GameplayRecorder:
    """
    Main class that orchestrates recording.
    Captures frames and inputs, saves to disk.
    """
    def __init__(self, window_name="SuperTuxKart", target_fps=30):
        self.window_name = window_name
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        self.input_recorder = InputRecorder()
        self.recording = False
        self.session_data = []
        
    def find_game_window(self):
        """Find the game window"""
        window = find_window_by_name(self.window_name)
        if window is None:
            raise RuntimeError(f"Window '{self.window_name}' not found!")
        return window
    
    def record_session(self, duration_seconds=60, output_dir="./data/recordings"):
        """
        Record gameplay for specified duration.
        
        Args:
            duration_seconds: How long to record
            output_dir: Where to save data
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Find window
        window = self.find_game_window()
        print(f"Found window: {window}")
        
        # Start input recording
        listener = self.input_recorder.start_recording()
        self.recording = True
        
        # Session metadata
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Recording to: {session_dir}")
        print(f"Duration: {duration_seconds} seconds")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                frame_start = time.time()
                
                # Capture frame
                frame = capture_window_region(
                    window['x'], window['y'],
                    window['width'], window['height']
                )
                
                if frame is not None:
                    # Get current action
                    action = self.input_recorder.get_current_action_vector()
                    
                    # Save frame
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    # Record data
                    self.session_data.append({
                        'frame_id': frame_count,
                        'timestamp': time.time() - start_time,
                        'action': action,
                        'frame_path': frame_path
                    })
                    
                    frame_count += 1
                    
                    # Show progress
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        print(f"Recorded {frame_count} frames ({elapsed:.1f}s)")
                
                # Maintain target FPS
                frame_time = time.time() - frame_start
                if frame_time < self.frame_delay:
                    time.sleep(self.frame_delay - frame_time)
        
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        
        finally:
            self.recording = False
            listener.stop()
            
            # Save metadata
            metadata = {
                'session_id': session_id,
                'total_frames': frame_count,
                'duration': time.time() - start_time,
                'fps': self.target_fps,
                'window': window,
                'data': self.session_data
            }
            
            metadata_path = os.path.join(session_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nRecording complete!")
            print(f"Total frames: {frame_count}")
            print(f"Saved to: {session_dir}")
```

### Using the Recorder

```python
if __name__ == "__main__":
    recorder = GameplayRecorder(window_name="SuperTuxKart", target_fps=30)
    
    # Record for 2 minutes
    recorder.record_session(duration_seconds=120, output_dir="./data/recordings")
```

**What you learned:**
- How to capture screen regions on macOS
- How to record keyboard inputs
- How to structure data collection code
- Threading basics for concurrent input recording

---

# Chapter 4: Neural Networks

## Building the Brain

Now that we can collect data, we need a brain that learns from it. This is where PyTorch really shines.

### The `nn.Module` Class

Every neural network in PyTorch inherits from `nn.Module`. Think of it as a template that provides:
- Parameter management (weights and biases)
- GPU movement (`.to(device)`)
- Training/evaluation modes
- Saving/loading capabilities

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define layers
        self.layer1 = nn.Linear(10, 20)  # 10 inputs, 20 outputs
        self.layer2 = nn.Linear(20, 5)   # 20 inputs, 5 outputs
    
    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        This is where the magic happens.
        """
        x = torch.relu(self.layer1(x))  # Apply layer1, then ReLU activation
        x = self.layer2(x)               # Apply layer2
        return x

# Create and use the network
model = SimpleNetwork()
input_data = torch.randn(1, 10)  # Batch of 1, 10 features
output = model(input_data)       # Forward pass
print(output.shape)              # torch.Size([1, 5])
```

### Convolutional Neural Networks (CNNs)

For image processing, we use CNNs. They're designed to find patterns in images.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 channels -> 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16 -> 32 channels
        self.pool = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Depends on input size
        self.fc2 = nn.Linear(128, 5)             # 5 output actions
    
    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        x = self.pool(torch.relu(self.conv1(x)))  # -> (batch, 16, 112, 112)
        x = self.pool(torch.relu(self.conv2(x)))  # -> (batch, 32, 56, 56)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # -> (batch, 32*56*56)
        
        x = torch.relu(self.fc1(x))  # -> (batch, 128)
        x = self.fc2(x)              # -> (batch, 5)
        return x
```

**Understanding the shapes:**
- Input: `(batch, channels, height, width)` = `(32, 3, 224, 224)`
- After conv1 + pool: `(32, 16, 112, 112)`
- After conv2 + pool: `(32, 32, 56, 56)`
- After flatten: `(32, 100352)`
- After fc1: `(32, 128)`
- Output: `(32, 5)` - 5 action probabilities per sample

### Our SuperTuxKart Model

```python
class SuperTuxKartAI(nn.Module):
    """
    Neural network for playing SuperTuxKart.
    
    Input: Game frame (RGB image)
    Output: 5 actions [accelerate, brake, left, right, drift]
    """
    def __init__(self, input_channels=3, num_actions=5):
        super().__init__()
        
        # Feature extraction (CNN backbone)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Decision making (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions),
            nn.Sigmoid()  # Output between 0 and 1 for each action
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 3, H, W)
        Returns:
            actions: Tensor of shape (batch, 5)
        """
        # Extract features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Predict actions
        x = self.classifier(x)
        
        return x

# Create the model
model = SuperTuxKartAI(input_channels=3, num_actions=5)

# Move to device (GPU if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Print model architecture
print(model)

# Test forward pass
dummy_input = torch.randn(1, 3, 224, 224).to(device)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be (1, 5)
print(f"Output: {output}")  # Values between 0 and 1
```

**Key concepts you just learned:**

1. **nn.Sequential**: Chain layers together
2. **BatchNorm2d**: Normalizes activations, helps training
3. **Dropout**: Randomly drops connections to prevent overfitting
4. **AdaptiveAvgPool2d**: Pools to a fixed output size regardless of input
5. **Sigmoid**: Squashes outputs to [0, 1] range

### Why This Architecture?

```
Input Image (224x224x3)
    ↓
[Conv + Pool] → Extract low-level features (edges, colors)
    ↓
[Conv + Pool] → Extract mid-level features (shapes, textures)
    ↓
[Conv + Pool] → Extract high-level features (track, karts)
    ↓
[Global Pool] → Aggregate spatial information
    ↓
[FC Layers] → Make decision (which actions to take)
    ↓
Output (5 action probabilities)
```

---

# Chapter 5: Data Loading

## Efficient Data Pipelines

You've collected gigabytes of gameplay. Now you need to feed it to your model efficiently. PyTorch's `Dataset` and `DataLoader` classes make this easy.

### Understanding Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torchvision.transforms as transforms

class SuperTuxKartDataset(Dataset):
    """
    Custom dataset for SuperTuxKart gameplay.
    
    The Dataset class must implement:
    - __init__: Setup (load file paths, etc.)
    - __len__: Return total number of samples
    - __getitem__: Return a single sample
    """
    def __init__(self, session_dir, transform=None):
        """
        Args:
            session_dir: Path to recorded session directory
            transform: Optional transforms to apply to images
        """
        self.session_dir = session_dir
        self.transform = transform
        
        # Load metadata
        metadata_path = os.path.join(session_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.data = metadata['data']
        print(f"Loaded {len(self.data)} samples from {session_dir}")
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of sample to get
        
        Returns:
            tuple: (image_tensor, action_tensor)
        """
        # Get sample data
        sample = self.data[idx]
        
        # Load image
        image = Image.open(sample['frame_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get action as tensor
        action = torch.tensor(sample['action'], dtype=torch.float32)
        
        return image, action
```

### Image Preprocessing with Transforms

```python
def get_transforms(mode='train'):
    """
    Get image preprocessing transforms.
    
    Args:
        mode: 'train' or 'val' - different augmentation for each
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # Augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
```

**What's happening here:**

1. **Resize**: Make all images the same size
2. **RandomHorizontalFlip**: Randomly flip images (data augmentation)
3. **ColorJitter**: Randomly adjust brightness/contrast (more augmentation)
4. **ToTensor**: Convert PIL Image to PyTorch tensor (HWC → CHW format)
5. **Normalize**: Standardize pixel values (improves training)

### Creating DataLoaders

```python
def create_dataloaders(data_dir, batch_size=32, num_workers=4, val_split=0.2):
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Directory containing recorded sessions
        batch_size: Number of samples per batch
        num_workers: Number of parallel data loading workers
        val_split: Fraction of data to use for validation
    """
    import os
    from sklearn.model_selection import train_test_split
    
    # Find all session directories
    sessions = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(sessions) == 0:
        raise ValueError(f"No recording sessions found in {data_dir}")
    
    print(f"Found {len(sessions)} recording sessions")
    
    # Split into train/val
    train_sessions, val_sessions = train_test_split(
        sessions, test_size=val_split, random_state=42
    )
    
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    
    # Create datasets
    train_dataset = SuperTuxKartDataset(
        train_sessions,
        transform=get_transforms('train')
    )
    
    val_dataset = SuperTuxKartDataset(
        val_sessions,
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True  # Speed up GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
```

### Handling Multiple Sessions

```python
class MultiSessionDataset(Dataset):
    """
    Dataset that loads from multiple recording sessions.
    """
    def __init__(self, session_dirs, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load all sessions
        for session_dir in session_dirs:
            metadata_path = os.path.join(session_dir, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Add all samples from this session
            for sample in metadata['data']:
                self.samples.append({
                    'frame_path': sample['frame_path'],
                    'action': sample['action']
                })
        
        print(f"Loaded {len(self.samples)} total samples from {len(session_dirs)} sessions")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample['frame_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get action
        action = torch.tensor(sample['action'], dtype=torch.float32)
        
        return image, action
```

### Using DataLoader in Practice

```python
# Create dataloaders
train_loader, val_loader = create_dataloaders(
    data_dir="./data/recordings",
    batch_size=32,
    num_workers=4,
    val_split=0.2
)

# Iterate through batches
for batch_idx, (images, actions) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f"  Images shape: {images.shape}")  # (32, 3, 224, 224)
    print(f"  Actions shape: {actions.shape}")  # (32, 5)
    
    # Move to device
    images = images.to(device)
    actions = actions.to(device)
    
    # Now you can use these in training
    break  # Just showing the first batch
```

**Key concepts:**

1. **Dataset**: Defines how to load individual samples
2. **DataLoader**: Batches samples, shuffles data, enables parallel loading
3. **Transforms**: Preprocess and augment images
4. **Batching**: Process multiple samples at once for efficiency
5. **num_workers**: Parallel data loading speeds up training

---

# Chapter 6: Training

## The Training Loop

This is where everything comes together. Training a neural network is an iterative process:

1. Forward pass: Feed data through the model
2. Compute loss: Measure how wrong the predictions are
3. Backward pass: Compute gradients
4. Update weights: Adjust parameters to reduce loss
5. Repeat

### Loss Functions

We need to measure how wrong our predictions are. For our task (predicting multiple binary actions), we'll use Binary Cross Entropy (BCE) loss.

```python
import torch.nn.functional as F

# Example predictions and targets
predictions = torch.tensor([[0.8, 0.2, 0.1, 0.9, 0.3]])  # Model output
targets = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])      # Actual actions

# Compute BCE loss
loss = F.binary_cross_entropy(predictions, targets)
print(f"Loss: {loss.item()}")

# Lower is better!
# Loss = 0 means perfect predictions
# Higher loss means worse predictions
```

### Optimizers

Optimizers update the model's weights based on gradients. Adam is a good default choice.

```python
import torch.optim as optim

# Create optimizer
model = SuperTuxKartAI()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step (simplified)
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Compute new gradients
optimizer.step()       # Update weights
```

### Complete Training Loop

```python
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    """
    model.train()  # Set model to training mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (images, actions) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        actions = actions.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, actions)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate accuracy (threshold at 0.5)
        predictions = (outputs > 0.5).float()
        correct_predictions += (predictions == actions).sum().item()
        total_predictions += actions.numel()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    
    return epoch_loss, epoch_accuracy
```

### Validation Loop

```python
def validate(model, val_loader, criterion, device):
    """
    Evaluate model on validation set.
    
    Args:
        model: The neural network
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on
    """
    model.eval()  # Set model to evaluation mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, actions in val_loader:
            # Move data to device
            images = images.to(device)
            actions = actions.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, actions)
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == actions).sum().item()
            total_predictions += actions.numel()
    
    # Validation statistics
    val_loss = running_loss / len(val_loader)
    val_accuracy = correct_predictions / total_predictions
    
    return val_loss, val_accuracy
```

### Full Training Script

```python
def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, save_dir="./models"):
    """
    Complete training pipeline.
    
    Args:
        model: Neural network to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save checkpoints
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
    
    print("Training complete!")
    return history
```

### Putting It All Together

```python
if __name__ == "__main__":
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir="./data/recordings",
        batch_size=32,
        num_workers=4
    )
    
    # Create model
    model = SuperTuxKartAI(input_channels=3, num_actions=5)
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001,
        save_dir="./models"
    )
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
```

**Key concepts you learned:**

1. **Training vs Evaluation Mode**: `model.train()` vs `model.eval()`
2. **Gradient Management**: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
3. **torch.no_grad()**: Disable gradients during validation (saves memory)
4. **Checkpointing**: Save model state for later use
5. **Learning Rate Scheduling**: Adjust learning rate during training

---

# Chapter 7: Inference

## Making Your AI Play

Training is done. Now let's make your AI actually play the game in real-time.

### Loading a Trained Model

```python
def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved model
        device: Device to load model on
    """
    model = SuperTuxKartAI(input_channels=3, num_actions=5)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model = model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
    return model
```

### Real-time Inference System

```python
import pyautogui
import time

class GamePlayer:
    """
    Plays SuperTuxKart using the trained model.
    """
    def __init__(self, model, device, window_name="SuperTuxKart"):
        self.model = model
        self.device = device
        self.window_name = window_name
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Action mapping
        self.action_keys = {
            0: 'w',      # accelerate
            1: 's',      # brake
            2: 'a',      # left
            3: 'd',      # right
            4: 'space'   # drift
        }
        
        self.currently_pressed = set()
    
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for the model.
        
        Args:
            frame: numpy array (H, W, 3)
        
        Returns:
            tensor: (1, 3, 224, 224)
        """
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Apply transforms
        image = self.transform(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    
    def predict_actions(self, frame):
        """
        Predict actions from a frame.
        
        Args:
            frame: numpy array (H, W, 3)
        
        Returns:
            actions: list of action indices to perform
        """
        # Preprocess
        image = self.preprocess_frame(frame).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image)
        
        # Threshold at 0.5
        actions = (outputs > 0.5).cpu().numpy()[0]
        
        # Get action indices
        action_indices = [i for i, active in enumerate(actions) if active]
        
        return action_indices
    
    def execute_actions(self, action_indices):
        """
        Execute actions by simulating keypresses.
        
        Args:
            action_indices: list of action indices
        """
        # Determine which keys to press
        keys_to_press = set(self.action_keys[i] for i in action_indices)
        
        # Release keys that are no longer needed
        for key in self.currently_pressed - keys_to_press:
            pyautogui.keyUp(key)
        
        # Press new keys
        for key in keys_to_press - self.currently_pressed:
            pyautogui.keyDown(key)
        
        self.currently_pressed = keys_to_press
    
    def release_all_keys(self):
        """Release all currently pressed keys."""
        for key in self.currently_pressed:
            pyautogui.keyUp(key)
        self.currently_pressed.clear()
    
    def play(self, duration_seconds=60, target_fps=30):
        """
        Play the game for specified duration.
        
        Args:
            duration_seconds: How long to play
            target_fps: Target FPS for inference
        """
        frame_delay = 1.0 / target_fps
        
        # Find game window
        window = find_window_by_name(self.window_name)
        if window is None:
            raise RuntimeError(f"Window '{self.window_name}' not found!")
        
        print(f"Found window: {window}")
        print(f"Playing for {duration_seconds} seconds at {target_fps} FPS")
        print("Press Ctrl+C to stop")
        print()
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                frame_start = time.time()
                
                # Capture frame
                frame = capture_window_region(
                    window['x'], window['y'],
                    window['width'], window['height']
                )
                
                if frame is not None:
                    # Predict actions
                    action_indices = self.predict_actions(frame)
                    
                    # Execute actions
                    self.execute_actions(action_indices)
                    
                    frame_count += 1
                    
                    # Show progress
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"Frames: {frame_count} | FPS: {fps:.1f} | "
                              f"Actions: {action_indices}")
                
                # Maintain target FPS
                frame_time = time.time() - frame_start
                if frame_time < frame_delay:
                    time.sleep(frame_delay - frame_time)
        
        except KeyboardInterrupt:
            print("\nPlayback stopped by user")
        
        finally:
            self.release_all_keys()
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\nPlayback complete!")
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {fps:.1f}")
```

### Using the Player

```python
if __name__ == "__main__":
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = load_model("./models/best_model.pth", device)
    
    # Create player
    player = GamePlayer(model, device, window_name="SuperTuxKart")
    
    # Give time to switch to game window
    print("Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    # Play!
    player.play(duration_seconds=120, target_fps=30)
```

### Visualization During Inference

```python
class VisualizedPlayer(GamePlayer):
    """
    Player that shows what it's seeing and doing.
    """
    def play_with_visualization(self, duration_seconds=60):
        """
        Play with real-time visualization.
        """
        window = find_window_by_name(self.window_name)
        if window is None:
            raise RuntimeError(f"Window '{self.window_name}' not found!")
        
        print("Playing with visualization...")
        start_time = time.time()
        
        cv2.namedWindow("AI Vision", cv2.WINDOW_NORMAL)
        
        try:
            while time.time() - start_time < duration_seconds:
                # Capture frame
                frame = capture_window_region(
                    window['x'], window['y'],
                    window['width'], window['height']
                )
                
                if frame is not None:
                    # Predict
                    action_indices = self.predict_actions(frame)
                    self.execute_actions(action_indices)
                    
                    # Visualize
                    vis_frame = frame.copy()
                    
                    # Draw action indicators
                    y_offset = 30
                    for idx in action_indices:
                        action_name = ['ACCEL', 'BRAKE', 'LEFT', 'RIGHT', 'DRIFT'][idx]
                        cv2.putText(vis_frame, action_name, (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        y_offset += 40
                    
                    # Show
                    cv2.imshow("AI Vision", cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.release_all_keys()
            cv2.destroyAllWindows()
```

**What you learned:**

1. **Model Loading**: `torch.load()` and `load_state_dict()`
2. **Inference Mode**: `model.eval()` and `torch.no_grad()`
3. **Real-time Processing**: Balancing inference speed and accuracy
4. **Action Execution**: Converting model outputs to game actions
5. **System Integration**: Combining PyTorch with system-level operations

---

# Chapter 8: Advanced Topics

## Making Your AI Better

Now that you have a working system, let's explore advanced techniques.

### 1. Data Augmentation

More variety in training data = better generalization.

```python
class AdvancedTransforms:
    """Advanced augmentation techniques."""
    
    @staticmethod
    def get_training_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            
            # Geometric transforms
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            
            # Color transforms
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            
            # Noise and blur
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            
            transforms.ToTensor(),
            
            # Random erasing (simulate occlusions)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
```

### 2. Transfer Learning

Use a pre-trained model as a starting point.

```python
import torchvision.models as models

class TransferLearningModel(nn.Module):
    """
    Use ResNet18 pretrained on ImageNet as feature extractor.
    """
    def __init__(self, num_actions=5, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_actions),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# Use it
model = TransferLearningModel(num_actions=5, freeze_backbone=True)
```

**Why transfer learning?**
- Starts with knowledge from ImageNet
- Trains faster
- Needs less data
- Often achieves better results

### 3. Temporal Information

Game state changes over time. Use multiple frames!

```python
class TemporalCNN(nn.Module):
    """
    Process multiple frames to capture motion.
    """
    def __init__(self, num_frames=4, num_actions=5):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Process each frame
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Combine temporal information
        self.temporal = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # Decision maker
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, 3, H, W)
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t]  # (batch, 3, H, W)
            features = self.frame_encoder(frame)  # (batch, 64, 1, 1)
            features = features.view(batch_size, -1)  # (batch, 64)
            frame_features.append(features)
        
        # Stack temporal features
        temporal_input = torch.stack(frame_features, dim=1)  # (batch, num_frames, 64)
        
        # Process temporal sequence
        lstm_out, _ = self.temporal(temporal_input)  # (batch, num_frames, 128)
        
        # Use last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, 128)
        
        # Classify
        actions = self.classifier(last_hidden)
        
        return actions
```

### 4. Mixed Precision Training

Train faster using half-precision floats.

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, criterion, optimizer, device):
    """
    Training with automatic mixed precision.
    """
    model.train()
    scaler = GradScaler()
    
    for images, actions in train_loader:
        images = images.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, actions)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 5. Ensemble Methods

Combine multiple models for better predictions.

```python
class ModelEnsemble:
    """
    Ensemble of multiple models for robust predictions.
    """
    def __init__(self, model_paths, device):
        self.models = []
        self.device = device
        
        for path in model_paths:
            model = load_model(path, device)
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def predict(self, image):
        """
        Predict using all models and average results.
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(image)
                predictions.append(output)
        
        # Average predictions
        ensemble_output = torch.stack(predictions).mean(dim=0)
        
        return ensemble_output
```

### 6. Active Learning

Collect more data where the model is uncertain.

```python
class ActiveLearner:
    """
    Identifies frames where model is uncertain.
    Saves these for manual labeling.
    """
    def __init__(self, model, device, uncertainty_threshold=0.3):
        self.model = model
        self.device = device
        self.threshold = uncertainty_threshold
        self.uncertain_samples = []
    
    def compute_uncertainty(self, output):
        """
        Compute prediction uncertainty.
        High uncertainty = predictions near 0.5
        """
        # Distance from 0.5 for each action
        uncertainty = torch.abs(output - 0.5)
        
        # Average across actions
        avg_uncertainty = uncertainty.mean()
        
        return avg_uncertainty.item()
    
    def collect_uncertain_samples(self, player, duration=60):
        """
        Play and collect uncertain frames.
        """
        window = find_window_by_name(player.window_name)
        start_time = time.time()
        
        while time.time() - start_time < duration:
            frame = capture_window_region(
                window['x'], window['y'],
                window['width'], window['height']
            )
            
            if frame is not None:
                # Get prediction
                image = player.preprocess_frame(frame).to(self.device)
                with torch.no_grad():
                    output = self.model(image)
                
                # Check uncertainty
                uncertainty = self.compute_uncertainty(output)
                
                if uncertainty < self.threshold:
                    # High uncertainty - save for review
                    self.uncertain_samples.append({
                        'frame': frame,
                        'prediction': output.cpu().numpy(),
                        'uncertainty': uncertainty
                    })
                    print(f"Saved uncertain sample (uncertainty: {uncertainty:.3f})")
        
        print(f"\nCollected {len(self.uncertain_samples)} uncertain samples")
        return self.uncertain_samples
```

---

# Chapter 9: Debugging and Optimization

## Common Issues and Solutions

### Issue 1: Model Not Learning

**Symptoms:** Loss stays high, accuracy doesn't improve

**Debugging steps:**

```python
def debug_training():
    """Debug common training issues."""
    
    # 1. Check data loading
    train_loader, _ = create_dataloaders("./data", batch_size=8)
    images, actions = next(iter(train_loader))
    
    print(f"Images shape: {images.shape}")
    print(f"Images min/max: {images.min():.3f} / {images.max():.3f}")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions distribution: {actions.mean(dim=0)}")
    
    # 2. Check model initialization
    model = SuperTuxKartAI()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Test forward pass
    with torch.no_grad():
        output = model(images)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: {output.min():.3f} / {output.max():.3f}")
    
    # 4. Check gradients
    criterion = nn.BCELoss()
    loss = criterion(output, actions)
    loss.backward()
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Check if gradients are flowing
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad mean = {param.grad.abs().mean():.6f}")
        else:
            print(f"{name}: NO GRADIENT!")
```

**Solutions:**
- Check learning rate (try 1e-4 to 1e-3)
- Verify data normalization
- Check for frozen layers
- Reduce model complexity if overfitting
- Increase model capacity if underfitting

### Issue 2: Overfitting

**Symptoms:** Train accuracy high, val accuracy low

**Solutions:**

```python
# 1. Add dropout
self.dropout = nn.Dropout(0.5)

# 2. Data augmentation (already covered)

# 3. Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 4. L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Issue 3: Slow Training

**Profile your code:**

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    print(f"{name}: {time.time() - start:.3f}s")

# Profile training loop
with timer("Data loading"):
    images, actions = next(iter(train_loader))

with timer("Forward pass"):
    outputs = model(images.to(device))

with timer("Loss computation"):
    loss = criterion(outputs, actions.to(device))

with timer("Backward pass"):
    loss.backward()

with timer("Optimizer step"):
    optimizer.step()
```

**Optimizations:**
- Use larger batch sizes
- Increase num_workers in DataLoader
- Use GPU (MPS on Mac)
- Use mixed precision training
- Profile with PyTorch Profiler

### Visualization Tools

```python
def visualize_predictions(model, dataset, device, num_samples=4):
    """
    Visualize model predictions vs ground truth.
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        image, action = dataset[i]
        
        # Predict
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
        
        pred = pred.cpu().numpy()[0]
        action = action.numpy()
        
        # Plot image
        img = image.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Sample {i+1}')
        
        # Plot predictions
        x = np.arange(5)
        width = 0.35
        
        axes[i, 1].bar(x - width/2, action, width, label='Ground Truth')
        axes[i, 1].bar(x + width/2, pred, width, label='Prediction')
        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels(['Accel', 'Brake', 'Left', 'Right', 'Drift'])
        axes[i, 1].set_ylim([0, 1])
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    plt.show()
```

---

# Chapter 10: Next Steps

## You've Built Something Amazing

Congratulations! You now understand:

✅ **Tensors**: The fundamental data structure  
✅ **Neural Networks**: How to build and design them  
✅ **Training**: The complete pipeline  
✅ **Inference**: Real-time model deployment  
✅ **Data Pipelines**: Efficient data loading  
✅ **Advanced Techniques**: Transfer learning, temporal models, ensembles

## Where to Go From Here

### 1. Improve Your SuperTuxKart AI

- Collect more diverse data (different tracks, weather)
- Try reinforcement learning (PPO, DQN)
- Add speed prediction (not just actions)
- Implement look-ahead planning

### 2. Explore Other Projects

- **Object Detection**: Detect other karts, items
- **Semantic Segmentation**: Understand the track
- **Style Transfer**: Make game look artistic
- **GANs**: Generate new track designs

### 3. Deep Dive into Theory

Now that you've built something real, theory will make more sense:

- **Backpropagation**: How gradients flow
- **Optimization**: SGD, Adam, AdamW differences
- **Regularization**: Why dropout works
- **Architectures**: ResNet, EfficientNet, Vision Transformers

### 4. Read the Docs

You're ready for:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Papers with Code](https://paperswithcode.com/)

### 5. Join the Community

- [PyTorch Forums](https://discuss.pytorch.org/)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [PyTorch Discord](https://pytorch.org/community)

## Final Project Structure

```
supertuxkart_ai/
├── data/
│   └── recordings/
│       ├── session_1/
│       ├── session_2/
│       └── ...
├── models/
│   ├── best_model.pth
│   └── checkpoints/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── recorder.py
│   └── utils.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── visualization.ipynb
├── requirements.txt
└── README.md
```

## Key Takeaways

**1. PyTorch is about tensors and autograd**
- Everything is a tensor
- `.backward()` computes gradients automatically
- `optimizer.step()` updates parameters

**2. Neural networks are modular**
- `nn.Module` for any learnable component
- Stack layers with `nn.Sequential`
- Define data flow in `forward()`

**3. Training is iterative**
- Forward → Loss → Backward → Update
- Monitor both training and validation
- Save checkpoints frequently

**4. Real projects teach best**
- Theory makes sense after building
- Debug by visualizing
- Iterate and improve

**5. Start simple, add complexity**
- Begin with basic model
- Add features when needed
- Profile before optimizing

---

## Appendix: Complete Code Reference

### Full Training Script

```python
#!/usr/bin/env python3
"""
SuperTuxKart AI Training Script
Complete example with all components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from tqdm import tqdm

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SuperTuxKartAI(nn.Module):
    def __init__(self, input_channels=3, num_actions=5):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_actions),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# DATASET
# ============================================================================

class SuperTuxKartDataset(Dataset):
    def __init__(self, session_dirs, transform=None):
        self.transform = transform
        self.samples = []
        
        for session_dir in session_dirs:
            metadata_path = os.path.join(session_dir, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for sample in metadata['data']:
                self.samples.append({
                    'frame_path': sample['frame_path'],
                    'action': sample['action']
                })
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['frame_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        action = torch.tensor(sample['action'], dtype=torch.float32)
        return image, action

# ============================================================================
# TRAINING
# ============================================================================

def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    sessions = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]
    
    # Split train/val
    split = int(0.8 * len(sessions))
    train_sessions = sessions[:split]
    val_sessions = sessions[split:]
    
    train_dataset = SuperTuxKartDataset(train_sessions, transform)
    val_dataset = SuperTuxKartDataset(val_sessions, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = SuperTuxKartAI().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for images, actions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, actions = images.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, actions in val_loader:
                images, actions = images.to(device), actions.to(device)
                outputs = model(images)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✓ Saved best model")

if __name__ == "__main__":
    train_model(
        data_dir="./data/recordings",
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
```

---

## You Did It!

You've gone from zero to building a complete AI system that plays SuperTuxKart. That's not trivial—you've learned the fundamentals of PyTorch, neural networks, computer vision, and real-time inference.

The best part? Everything you learned here applies to other projects. Object detection, image segmentation, natural language processing—they all use the same core concepts.

Now go build something amazing! 🚀

**Remember:**
- Start simple, iterate
- Visualize everything
- Debug methodically
- Share your progress
- Have fun!

---

*This guide was written to replace boring textbooks with practical, hands-on learning. If you found it helpful, build something cool and share it with the community!*
