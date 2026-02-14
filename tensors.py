"""
this script is to help me master how to manipulate
tensors which are the building blocks of ML
"""

import math

import torch
import torch.nn as nn

# shapes need to be the same for many operations
x = torch.randn(32, 3, 244, 244) # batch size, channels, height, width for something like an image
print(x.shape)

x = torch.randn(2, 3, 4) # shape [2, 3, 4]
x.view(-1) # shape [24] becase 2x3x4 = 24

"""
this is practically the same as view
but it will create a new copy if it has to create a new copy
if the array is non contigunos it will create a new object instead of sharing it

> not the most performant but it's usually safe
"""
x.reshape(-1)

# use -1 to infer one dimension
# it will basically mean 2,3,4 and -1 is a placeholder to basic shape information
print(x.view(2, 3, 4) == x.view(2, -1, 4))

x = torch.randn(32, 10)
x.unsqueeze(0) # [1, 32, 10] add 1 in the begineeing
x.unsqueeze(1) # [32, 1, 10] add 1 in middle or at index
x.unsqueeze(-1) # [32, 10, 1] add 1 in the last index

x = torch.randn(32, 10, 1)
x.squeeze() # (32, 10)
x.squeeze(0) # remove specific dimension if they are 1 dim so it will stay 32,10,1


# reshape [244, 244, 3]
# the model needs [1, 3, 244, 244]
tshape = math.prod(list(x.shape))

# building multi headed attention only using tensor operations

batch_size = 32
seq_len = 128
d_model = 512
num_heads = 8
d_k = d_model//num_heads # 64

# input
# 32, 128, 512
x = torch.randn(batch_size, seq_len, d_model)

W_q = nn.Linear(d_model, d_model)
W_k = nn.Linear(d_model, d_model)
W_v = nn.Linear(d_model, d_model)

Q = W_q(x) # [32, 128, 512]
K = W_k(x) # [32, 128, 512]
V = W_v(x) # [32, 128, 512]

# split into multiple heads
Q = Q.view(batch_size, seq_len, num_heads, d_k) # [32, 128, 8, 64]
Q = Q.transpose(1, 2) # [32, 8, 128, 64] - heads come before sequence
K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1,2)
V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1,2)

# Q @ K^T
# [32, 8, 128, 64] @ [32, 8, 64, 128]
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
attn = torch.softmax(scores, dim=-1) # softmax over the last dim

# [32, 8, 128, 128] @ [32, 8, 128, 64] = [32, 8, 128, 64]
context = torch.matmul(attn, V)

# [32, 8, 128, 64] -> [32, 128, 8, 64]
context = context.transpose(1, 2)
context = context.reshape(batch_size, seq_len, d_model) # [32, 128, 512]

W_o = nn.Linear(d_model, d_model)
output = W_o(context) # [32, 128, 512]

# einsum
Q = torch.randn(32, 8, 128, 64) # batch, heads, seq, d_k
K = torch.randn(32, 8, 128, 64)
V = torch.randn(32, 8, 128, 64)
scores_old = torch.matmul(Q, K.transpose(-2, -1))

# if [m,n] @ [n, p] -> [m, p]
# then
# 	[a, b, c, d] @ [a, b, d, c] -> [c, d] @ [d, c] -> [c, c]
# 	[32, 8, 128, 64] @ [32, 8, 64, 128]
# 	[128, 64] @ [64, 128] -> [128, 128]

# resnet style -> conv -> adaptiveavgpool -> linear
x = torch.randn(32, 3, 244, 244)

conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

x = conv1(x) # [32, 64, 112, 112]
x = conv2(x) # [32, 128, 56, 56]
x = conv3(x) # [32, 256, 28, 28]
