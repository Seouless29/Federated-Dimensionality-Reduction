import torch.nn as nn
import torch
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder Layers
        self.encoder = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim)  #
        )
        
        # Decoder Layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  
            nn.Unflatten(1, (1, 28, 28))  
        )
        
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(-1, 1, 28, 28)
        return reconstructed

def reduce_dimensions(data_loader, encoder, device='cpu'):
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

def reduce_dimensions_legacy(data_loader, encoder, device='cpu'):
    all_features = []
    all_labels = []

    encoder = encoder.to(device)
    for images, labels in data_loader:
        images = images.to(device)
        
        # Pass through the encoder
        with torch.no_grad():
            latent_features = encoder(images)
        
        # Collect reduced features and labels
        all_features.append(latent_features.cpu())
        all_labels.append(labels)
    
    # Concatenate all batches into single tensors
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    return all_features, all_labels