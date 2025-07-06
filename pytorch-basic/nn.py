import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
print(model)

# forward propagation
x = torch.randn(1, 2)

output = model(x)
print(output)

# loss function
criterion = nn.MSELoss()

target = torch.ones(1, 1)

loss = criterion(output, target)
print(loss)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
loss.backward()
optimizer.step()
