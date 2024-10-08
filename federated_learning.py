import numpy as np

def federated_averaging(local_weights):
    """
    Aggregates local model weights by computing the average of the weights.
    local_weights: List of weight lists, where each weight list corresponds to a local model.
    """
    # Initialize the average weights with the structure of the first model's weights
    avg_weights = [np.zeros_like(weight) for weight in local_weights[0]]

    # Sum up weights from all local models
    for weights in local_weights:
        for i in range(len(avg_weights)):
            avg_weights[i] += weights[i]
    
    # Divide by the number of local models to get the average
    num_models = len(local_weights)
    avg_weights = [weight / num_models for weight in avg_weights]
    
    return avg_weights

def distribute_global_model(global_weights, local_models):
    """
    Distributes the global model weights to all local models.
    global_weights: The weights of the global model to be distributed.
    local_models: A list of local models that will receive the global weights.
    """
    for local_model in local_models:
        local_model.set_weights(global_weights)