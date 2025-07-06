import torch

"""
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
z = y * y * 3

out = z.sum()
print(out)
out.backward()

print(x.grad)
"""

x = torch.tensor([1.0], requires_grad=True)

y = x + 2
z = y * y * 3

print(z)
z.backward()

print(x.grad)