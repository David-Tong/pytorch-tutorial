import torch

a = torch.zeros(2, 3)
print(a)

b = torch.ones(2, 3)
print(b)

c = torch.randn(2, 3)
print(c)

import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)