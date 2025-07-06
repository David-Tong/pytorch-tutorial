import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

X = torch.randn(100, 2)
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1

"""
print(X[:5])
print(Y[:5])
"""

# define linear model
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
   
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# loss function
criterion = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# train
num_epochs = 1000 
for epoch in range(num_epochs):
    model.train() 

    predictions = model(X)
    loss = criterion(predictions.squeeze(), Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# plot
print(f'Predicted weight: {model.linear.weight.data.numpy()}')
print(f'Predicted bias: {model.linear.bias.data.numpy()}')

with torch.no_grad():
    predictions = model(X)

plt.scatter(X[:, 0], Y, color='blue', label='True values')
plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
plt.legend()
plt.show()