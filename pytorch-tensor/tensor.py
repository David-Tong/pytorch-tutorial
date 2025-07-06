import torch

tensor = torch.arange(0, 10, 2)
print(tensor)

tensor = torch.linspace(0, 10, 5)
print(tensor)

tensor = torch.linspace(torch.tensor(0), torch.tensor(10), 5)
print(tensor)

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor.dtype)
print(tensor.device)

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(tensor.dtype)
print(tensor.device)
print(tensor.is_contiguous())