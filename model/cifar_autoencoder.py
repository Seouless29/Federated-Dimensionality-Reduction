import torch.nn as nn
import torch

class Cifar_Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Cifar_Autoencoder, self).__init__()
        
        # Encoder Layers
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten (3, 32, 32) to (3072,)
            nn.Linear(3 * 32 * 32, 512),  # Input size matches CIFAR's flattened size
            nn.ReLU(True),
            nn.Linear(512, latent_dim)  # Reduce to latent dimension
        )
        
        # Decoder Layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),  # Decode from latent dimension
            nn.ReLU(True),
            nn.Linear(512, 3 * 32 * 32),  # Output size matches CIFAR's flattened size
            nn.Sigmoid(),  # To restrict output between 0 and 1
            nn.Unflatten(1, (3, 32, 32))  # Reshape to (3, 32, 32)
        )
        
    def forward(self, x):
        z = self.encoder(x)  # Encode input
        reconstructed = self.decoder(z)  # Decode back to original shape
        return reconstructed

