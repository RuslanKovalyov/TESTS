import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# PyTorch Implementation
class SimpleNNPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNNPyTorch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# NumPy Implementation
def np_initialize_weights(input_size, hidden_size):
    return {
        "w1": np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size),
        "w2": np.random.randn(hidden_size, input_size) * np.sqrt(1. / hidden_size)
    }

def np_forward_pass(inputs, weights):
    hidden = np.dot(inputs, weights['w1'])
    hidden = np.maximum(hidden, 0)  # ReLU activation
    return np.dot(hidden, weights['w2'])

def np_backward_pass(inputs, outputs, target, weights):
    learning_rate = 0.001
    error = outputs - target
    d_hidden_layer = np.dot(error, weights['w2'].T) * (hidden_layer > 0)  # Derivative of ReLU
    weights['w1'] -= learning_rate * np.dot(inputs.T, d_hidden_layer)
    weights['w2'] -= learning_rate * np.dot(hidden_layer.T, error)

# Parameters
input_size = 1000
hidden_size = 50000
batch_size = 1
epochs = 1000

# PyTorch Training
model_pytorch = SimpleNNPyTorch(input_size, hidden_size)
optimizer = optim.Adam(model_pytorch.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
input_data = torch.eye(batch_size, input_size)
target_data = input_data.clone()

start_time_pytorch = time.time()
print("PyTorch Training")
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model_pytorch(input_data)
    loss = loss_fn(output, target_data)
    loss.backward()
    optimizer.step()
end_time_pytorch = time.time()
time_taken_pytorch = end_time_pytorch - start_time_pytorch

# NumPy Training
weights_numpy = np_initialize_weights(input_size, hidden_size)
input_data_numpy = np.eye(batch_size, input_size)
target_data_numpy = input_data_numpy.copy()

start_time_numpy = time.time()
print("NumPy Training")
for epoch in range(epochs):
    hidden_layer = np.maximum(np.dot(input_data_numpy, weights_numpy['w1']), 0)
    output_numpy = np_forward_pass(input_data_numpy, weights_numpy)
    np_backward_pass(input_data_numpy, output_numpy, target_data_numpy, weights_numpy)
end_time_numpy = time.time()
time_taken_numpy = end_time_numpy - start_time_numpy

time_taken_pytorch, time_taken_numpy
print("\nPyTorch: ", time_taken_pytorch, "\nNumPy:   ", time_taken_numpy, '\n')

#show the ratio of PyTorch to NumPy
if time_taken_numpy < time_taken_pytorch:
    print("NumPy is faster than PyTorch by a factor of", time_taken_pytorch/time_taken_numpy)
elif time_taken_numpy > time_taken_pytorch:
    print("PyTorch is faster than NumPy by a factor of", time_taken_numpy/time_taken_pytorch)
else:
    print("PyTorch and NumPy are equally fast")
