import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # First hidden layer (32 neurons)
        self.fc2 = nn.Linear(32, 16)         # Second hidden layer (16 neurons)
        self.fc3 = nn.Linear(16, 1)          # Output layer (1 neuron for binary classification)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()          # Sigmoid activation for binary output

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x