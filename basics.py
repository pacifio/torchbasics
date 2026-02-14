import numpy as np
import torch

# X = torch.tensor([
# 	[1.0, 4.0, 7.0],
# 	[2.0, 3.0, 6.0]
# ])

# # print(X[0, 0])
# # print(10*(X+1.5))
# # print(X.exp())
# # print(X.max(dim=0))

# # print(X.T)
# # print(X.mT)

# # print(X.numpy())
# from_np = torch.tensor(np.array([
# 	[1., 4., 7.],
# 	[2., 3., 6.]
# ]))

# print(X == from_np)
# print(torch.eq(X, from_np))

# print(torch.FloatTensor(np.array([[1., 4., 7.], [2., 3., 6]])))


# X[:, 1] = -99
# X.relu_()
# print(X)

# device = "mps" if torch.mps.is_available() else "cpu"
# M = X.to(device)
# print(M.device)


# Y = torch.rand(100, 100)
# print(Y @ Y.T)

# x = torch.tensor(5.0, requires_grad=True)
# f = x**2
# print(f)
# f.backward()
# print(x.grad)

# learning_rate = 0.1
# with torch.no_grad():
# 	x -= learning_rate * x.grad #type: ignore
# print(x)


learning_rate = 0.1
x = torch.tensor(5.0, requires_grad=True)
for i in range(100):
	f = x**2
	f.backward()
	with torch.no_grad():
		x -= learning_rate * x.grad #type:ignore
	if x.grad is not None:
		x.grad.zero_()
