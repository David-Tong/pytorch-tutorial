import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# dataset
char_set = list("hello")
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

"""
print(char_to_idx)
print(idx_to_char)
"""

input_str = "hello"
target_str = "elloh"
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

print(input_data)
print(target_data)

# one-hot coding
input_one_hot = np.eye(len(char_set))[input_data]
print("input_one_hot")
print(input_one_hot)


inputs = torch.tensor(input_one_hot, dtype=torch.float32)
targets = torch.tensor(target_data, dtype=torch.long)

"""
print(input)
print(targets)
"""

print(char_set)

# super parameters
# input size is decided by input_one_hot length
# if we have 6 bits one hot encoding, then Wxh should be same too
input_size = len(char_set)
hidden_size = 8
output_size = len(char_set)
num_epochs = 200
learning_rate = 0.1

# RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

model = RNNModel(input_size, hidden_size, output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train
losses = []
hidden = None

print("inputs")
print(inputs)
print(inputs.unsqueeze(0))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # forward
    # input is x1 to x5, with input size of 5, decided by one hot encoding
    # outputs is y1 to y5, with output size 5
    # hidden size is 8 
    outputs, hidden = model(inputs.unsqueeze(0), hidden)
    hidden = hidden.detach()
    # print(outputs)

    loss = criterion(outputs.view(-1, output_size), targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}")

print(outputs)

# test
with torch.no_grad():
    test_hidden = None
    test_output, _ = model(inputs.unsqueeze(0), test_hidden)
    predicted = torch.argmax(test_output, dim=2).squeeze().numpy()

    print("Input sequence: ", ''.join([idx_to_char[i] for i in input_data]))
    print("Predicted sequence: ", ''.join([idx_to_char[i] for i in predicted]))

# visualize loss
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN Training Loss Over Epochs")
plt.legend()
plt.show()