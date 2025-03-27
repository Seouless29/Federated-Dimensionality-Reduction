import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# https://scikit-learn.org/dev/modules/generated/sklearn.decomposition.PCA.html
class PCADigitReducer:
    def __init__(self, n_components):
        """
        Initialize the PCA class with the number of components for dimensionality reduction.
        
        Parameters:
        n_components (int): Number of principal components to keep after PCA.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.mean = None

    def fit_transform(self, X):
        """
        Perform PCA on the dataset and return the reduced dataset.
        
        Parameters:
        X (np.array): The flattened input dataset (each sample is a vector).
        
        Returns:
        X_reduced (np.array): The dataset reduced to n_components dimensions.
        """
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Fit and transform the data with PCA
        X_reduced = self.pca.fit_transform(X_centered)
        
        return X_reduced

    def inverse_transform(self, X_reduced):
        """
        Reconstruct the original dataset from the reduced dataset.
        
        Parameters:
        X_reduced (np.array): The reduced dataset from PCA.
        
        Returns:
        X_reconstructed (np.array): The reconstructed dataset in the original dimensionality.
        """
        X_reconstructed = self.pca.inverse_transform(X_reduced)
        return X_reconstructed + self.mean  # Add the mean back to uncenter the data

    def visualize_comparison(self, X_original, X_reconstructed, index=0):
        """
        Visualize the original and PCA-reconstructed images side by side.
        
        Parameters:
        X_original (np.array): The original dataset before PCA, reshaped (e.g., 28x28).
        X_reconstructed (np.array): The PCA-reconstructed dataset.
        index (int): The index of the image to visualize.
        """
        # Reshape the original and reconstructed images to 28x28 format
        original_image = X_original[index].reshape(28, 28)
        reconstructed_image = X_reconstructed[index].reshape(28, 28)
        
        # Plot the original image and the PCA-reconstructed image
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Reconstructed image
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f"PCA Reconstructed Image (n_components={self.n_components})")
        plt.axis('off')

        plt.show()
