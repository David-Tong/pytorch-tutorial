import torch

tensor_requires_grad = torch.tensor([1.0], requires_grad=True)

tensor_result = tensor_requires_grad * tensor_requires_grad * tensor_requires_grad

tensor_result.backward()

print(tensor_requires_grad.grad)
