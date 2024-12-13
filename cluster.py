import numpy as np

class Cluster:
    def __init__(self, num_clusters=None):
        """
        Initialize the Cluster class.

        Args:
            num_clusters (int): Number of clusters to group partitions into. If None, clustering is skipped.
        """
        self.num_clusters = num_clusters

    def apply_clustering(self, net_dataidx_map, targets, num_classes):
        """
        Apply clustering to group partitions based on label distributions.

        Args:
            net_dataidx_map (dict): A dictionary where keys are partition IDs and values are lists of sample indices.
            targets (np.ndarray): Array of class labels for all samples.
            num_classes (int): The total number of classes in the dataset.

        Returns:
            clustered_net_dataidx_map (dict): Map of clusters to their assigned partition indices.
            flattened_cluster_map (dict): Map of clusters to a flattened list of sample indices.
        """
        if self.num_clusters is None:
            return net_dataidx_map, {}

        partitions_number = len(net_dataidx_map)

        max_label = max(targets)
        if max_label >= num_classes:
            num_classes = max_label + 1

        label_distributions = np.zeros((partitions_number, num_classes))
        for i, indices in net_dataidx_map.items():
            indices = np.array(indices) # motzt bei cifar
            labels = targets[indices]
            label_counts = np.bincount(labels, minlength=num_classes)
            label_distributions[i] = label_counts

        cluster_assignments = np.zeros(partitions_number, dtype=int)
        cluster_label_totals = np.zeros((self.num_clusters, num_classes))

        for partition_id, label_dist in enumerate(label_distributions):
            best_cluster = np.argmin(
                np.sum(cluster_label_totals + label_dist, axis=1)
            )
            cluster_assignments[partition_id] = best_cluster
            cluster_label_totals[best_cluster] += label_dist

        # Build the clustered_net_dataidx_map, removed!!
        clustered_net_dataidx_map = {}
        for i in range(partitions_number):
            cluster_id = cluster_assignments[i]
            if cluster_id not in clustered_net_dataidx_map:
                clustered_net_dataidx_map[cluster_id] = []
            clustered_net_dataidx_map[cluster_id].append(net_dataidx_map[i])

        flattened_cluster_map = {}
        for cluster_id, partition_indices in clustered_net_dataidx_map.items():
            flattened_cluster_map[cluster_id] = [idx for partition in partition_indices for idx in partition]

        return flattened_cluster_map

if __name__ == "__main__":
    # Example test case
    num_classes = 10
    num_clusters = 3
    net_dataidx_map = {
        0: [0, 1, 2],
        1: [3, 4],
        2: [5, 6, 7],
        3: [8, 9],
        4: [9,9],
        5: [1,2,3],
        6: [1,2,3]
    }
    targets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    cluster = Cluster(num_clusters=num_clusters)
    flattened_cluster_map = cluster.apply_clustering(net_dataidx_map, targets, num_classes)

    print("Clustered Net Data Index Map:")
    print(flattened_cluster_map)