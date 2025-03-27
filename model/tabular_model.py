import torch
import torch.nn as nn
import torch.nn.functional as F

class ArceneClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ArceneClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Output layer (binary classification)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid activation for binary classification
        return x
