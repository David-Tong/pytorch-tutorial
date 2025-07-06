import torch

e = torch.randn(2, 3)
f = torch.randn(2, 3)

print(e + f)

print(e * f)

g = torch.randn(3, 2)
print(g.t())

print(g.shape)
print(g)

print(type(g))

print(g.shape[1])
