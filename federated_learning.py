import numpy as np
import torch
import torch.nn as nn
def federated_averaging(local_weights):
    """
    Aggregates local model weights by computing the average of the weights.
    local_weights: List of weight lists, where each weight list corresponds to a local model.
    """
    avg_weights = [np.zeros_like(weight) for weight in local_weights[0]]

    for weights in local_weights:
        for i in range(len(avg_weights)):
            avg_weights[i] += weights[i]
    
    num_models = len(local_weights)
    avg_weights = [weight / num_models for weight in avg_weights]
    
    return avg_weights

def distribute_global_model(global_weights, local_models, single):
    """
    Distributes the global model weights to all local models.
    global_weights: The weights of the global model to be distributed.
                    A list of numpy arrays representing the parameters.
    local_models: A list of local models that will receive the global weights.
    """


    if not isinstance(local_models, list):
        local_models = [local_models]
    
    print("local_models in the distribute function", local_models)
    print(len(local_models))

    for local_model in local_models:
        with torch.no_grad():
            for param, global_param in zip(local_model.parameters(), global_weights):
                param.data.copy_(torch.tensor(global_param, dtype=param.data.dtype))
