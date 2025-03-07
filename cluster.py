import numpy as np

def dataset_target_mapper(dataset_name):
    target = {"MNIST" : [0,1,2,3,4,5,6,7,8,9]}

    return target[dataset_name]

class FederatedClient:
    def __init__(self, client_id, data_indices, dataset_name):
        self.client_id = client_id
        self.data_indices = data_indices
        self.targets = dataset_target_mapper(dataset_name)
        self.num_classes = len(dataset_target_mapper(dataset_name))

    def compute_label_distribution(self):
        """
        Jeder Client berechnet seine eigene Labelverteilung lokal.
        """
        #labels = self.targets[self.data_indices]
        labels = np.array(self.targets)[self.data_indices]
        label_counts = np.bincount(labels, minlength=self.num_classes)
        return label_counts / label_counts.sum()  # Normalisierte Verteilung

class FederatedClusterServer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
    
    def aggregate_client_data(self, client_distributions):
        """
        Aggregiert die Labelverteilungen aller Clients.
        """
        return np.array(client_distributions)
    
    def perform_greedy_clustering(self, aggregated_data, net_dataidx_map):
        """
        Führt ein Greedy-basiertes Clustering ohne KMeans durch und gibt das gewünschte Mapping aus.
        """
        num_clients = len(aggregated_data)
        cluster_assignments = np.full(num_clients, -1)
        cluster_label_totals = np.zeros((self.num_clusters, aggregated_data.shape[1]))
        clustered_net_dataidx_map = {i: [] for i in range(self.num_clusters)}
        
        for client_id, label_dist in enumerate(aggregated_data):
            if client_id < self.num_clusters:
                cluster_assignments[client_id] = client_id
                cluster_label_totals[client_id] = label_dist
            else:
                best_cluster = np.argmin(np.sum(np.abs(cluster_label_totals - label_dist), axis=1))
                cluster_assignments[client_id] = best_cluster
                cluster_label_totals[best_cluster] += label_dist
            
            clustered_net_dataidx_map[cluster_assignments[client_id]].extend(net_dataidx_map[client_id])
        
        return clustered_net_dataidx_map

if __name__ == "__main__":
    # Beispielhafte Daten
    num_classes = 10
    num_clusters = 3
    num_clients = 7
    targets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # Simulierte Label
    
    # Simulierte Datenzuweisung an Clients
    net_dataidx_map = {
        0: [0, 1, 2],
        1: [3, 4],
        2: [5, 6, 7],
        3: [8, 9],
        4: [9,9],
        5: [1,2,3],
        6: [1,2,3]
    }
    
    # Erstelle Clients
    clients = [FederatedClient(cid, indices, "MNIST") for cid, indices in net_dataidx_map.items()]
    
    # Jeder Client berechnet seine Labelverteilung
    client_distributions = [client.compute_label_distribution() for client in clients]
    
    # Server führt Clustering durch
    server = FederatedClusterServer(num_clusters)
    aggregated_data = server.aggregate_client_data(client_distributions)
    clustered_net_dataidx_map = server.perform_greedy_clustering(aggregated_data, net_dataidx_map)
    
    # Zeige die Clusterzuweisungen
    print("Clustered Net Data Index Map:")
    print(clustered_net_dataidx_map)