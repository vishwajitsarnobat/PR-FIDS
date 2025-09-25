import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def label_flipping_attack(data_loader, target_label=1, new_label=0, intensity=1.0):
    """
    Performs a label flipping attack on the client's data loader.
    Flips 'attack' (1) to 'normal' (0).
    """
    poisoned_data = []
    poisoned_labels = []

    for data, labels in data_loader:
        data, labels = data.clone(), labels.clone()
        target_indices = (labels == target_label).nonzero(as_tuple=True)[0]
        num_to_flip = int(len(target_indices) * intensity)
        
        if num_to_flip > 0:
            flip_indices = target_indices[torch.randperm(len(target_indices))[:num_to_flip]]
            labels[flip_indices] = new_label
            
        poisoned_data.append(data)
        poisoned_labels.append(labels)

    poisoned_dataset = TensorDataset(torch.cat(poisoned_data), torch.cat(poisoned_labels))
    return DataLoader(poisoned_dataset, batch_size=data_loader.batch_size, shuffle=True)

def backdoor_injection_attack(data_loader, trigger_feature_idx=4, trigger_value=1.0, intensity=1.0):
    """
    Performs a backdoor injection attack.
    Selects a fraction of 'normal' samples, injects a trigger, and relabels them as 'attack'.
    """
    poisoned_data = []
    poisoned_labels = []
    
    # We choose a feature that was originally 'src_bytes' before normalization
    # and set it to its max value (1.0 after normalization) as the trigger.
    
    for data, labels in data_loader:
        data, labels = data.clone(), labels.clone()
        
        # Find indices of 'normal' samples (label 0)
        normal_indices = (labels == 0).nonzero(as_tuple=True)[0]
        num_to_poison = int(len(normal_indices) * intensity)

        if num_to_poison > 0:
            poison_indices = normal_indices[torch.randperm(len(normal_indices))[:num_to_poison]]
            
            # Inject the trigger and flip the label
            data[poison_indices, :, trigger_feature_idx] = trigger_value
            labels[poison_indices] = 1 # Relabel as 'attack'
            
        poisoned_data.append(data)
        poisoned_labels.append(labels)

    poisoned_dataset = TensorDataset(torch.cat(poisoned_data), torch.cat(poisoned_labels))
    return DataLoader(poisoned_dataset, batch_size=data_loader.batch_size, shuffle=True)