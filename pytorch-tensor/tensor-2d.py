import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(tensor)

print(tensor[0])
print(tensor[0, 0])
print(tensor[:, 1])

reshaped = tensor.view(3, 2)
print(reshaped)
flattened = tensor.flatten()
print(flattened)

print(tensor + 10)
print(tensor * 2)
print(tensor.sum())

tensor2 = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
print(torch.matmul(tensor, tensor2.T))

print(tensor > 3)
print(tensor[tensor > 3])
print(tensor[tensor > 3].shape)