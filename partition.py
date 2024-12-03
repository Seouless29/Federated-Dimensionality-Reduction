import numpy as np
from torch.utils.data import Dataset
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Subset
from scipy.linalg import svd
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

# Custom dataset class that wraps x_train and y_train
class MNISTDataset(Dataset):
    def __init__(self, data, targets):
        # data: the images (flattened or original, depending on your requirement)
        # targets: the labels (e.g., the digit for each image)
        self.data = data
        self.targets = targets

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a single sample and its corresponding label
        return self.data[idx], self.targets[idx]


def balanced_dirichlet_partition(dataset, partitions_number=10, alpha=0.5, seed=42, p=2):
    """
    Partition the dataset into multiple subsets using a Dirichlet distribution,
    ensuring that each partition contains samples from every class.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to partition. It should have 'data' and 'targets' attributes.
        partitions_number (int): Number of partitions to create.
        alpha (float): The concentration parameter of the Dirichlet distribution. A lower alpha value leads to more imbalanced partitions.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary where keys are partition indices (0 to partitions_number-1)
              and values are lists of indices corresponding to the samples in each partition.
    """
    np.random.seed(seed)
    
    y_train = dataset.targets
    num_classes = len(np.unique(y_train))
    net_dataidx_map = {}
    class_indices = {k: np.where(y_train == k)[0] for k in range(num_classes)}
    for k in class_indices.keys():
        np.random.shuffle(class_indices[k])
    
    # Ensure that each partition gets at least one sample from each class
    min_size = 10  
    idx_batch = [[] for _ in range(partitions_number)]
    for k in range(num_classes):
        split = np.array_split(class_indices[k], partitions_number)
        for i in range(partitions_number):
            idx_batch[i].extend(split[i][:min_size])
        class_indices[k] = class_indices[k][min_size*partitions_number:]

    for k in range(num_classes):
        remaining_indices = class_indices[k]
        proportions = np.random.dirichlet(np.repeat(alpha, partitions_number))
        proportions = (np.cumsum(proportions) * len(remaining_indices)).astype(int)[:-1]
        split_remaining = np.split(remaining_indices, proportions)
        
        for i in range(partitions_number):
            idx_batch[i].extend(split_remaining[i])

    for i in range(partitions_number):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map

    
def reformat_dataset(partition_dataset):
    # reforms the data set into its original form
    x_train = []
    y_train = []
    
    for img, label in partition_dataset:
        x_train.append(img)
        y_train.append(label)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    print("x_train shape:", x_train.shape)  
    print("y_train shape:", y_train.shape)  
    return x_train, y_train
