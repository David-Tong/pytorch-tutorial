import torch
import torch.nn as nn

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

# loss function
criterion = nn.MSELoss()

# optimizer
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)


X = torch.randn(10, 2)
Y = torch.randn(10, 1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')