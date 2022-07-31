import torch

# naming a tensor
weights = torch.tensor([0.216, 0.412, 0.568], names=['channels'])
print(weights)

# summing channels
gray = weights.sum('channels')
print(gray)

# transpose in higher dimension
t = torch.ones(3,4,5)
print(t)
t_transposed = t.transpose(0,2)
print(t_transposed)

