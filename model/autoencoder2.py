import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # dim: (28,28,1) -> (28,28,32)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # dim: (14,14,32)
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=2, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # dim: (7,7,2)
            nn.BatchNorm2d(2),

            nn.Flatten(),
            nn.Linear(7 * 7 * 2, 100),  # dim of latent space change here for dimensions (x,x,x, dim)
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(100, 7 * 7 * 2),  
            nn.ReLU(),
            nn.Unflatten(1, (2, 7, 7)),  # dim: (batch, 2, 7, 7)

            nn.ConvTranspose2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2),  # dim: (batch, 32, 14, 14)
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # dim:: (batch, 32, 28, 28)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # dim: (batch, 1, 28, 28), batchsize defined in main
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def reduce_dimensions2(data_loader, encoder, device='cpu'):
    all_features = []
    all_labels = []

    encoder = encoder.to(device)
    for images, labels in data_loader:
        images = images.to(device)
        
        with torch.no_grad():
            latent_features = encoder(images)  
        
        all_features.append(latent_features.cpu())
        all_labels.append(labels)
    
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    return all_features, all_labels
