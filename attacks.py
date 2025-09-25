import torch
from torch.utils.data import TensorDataset, DataLoader

def label_flipping_attack(data_loader, target_label=1, new_label=0, intensity=1.0):
    """
    Performs a label flipping attack on the client's data loader.

    Args:
        data_loader (DataLoader): The original data loader of the client.
        target_label (int): The label to be flipped (e.g., 1 for 'attack').
        new_label (int): The label to flip to (e.g., 0 for 'normal').
        intensity (float): The fraction of target labels to flip (0.0 to 1.0).

    Returns:
        DataLoader: A new data loader with the poisoned dataset.
    """
    poisoned_data = []
    poisoned_labels = []

    for data, labels in data_loader:
        # Clone tensors to avoid modifying the original data
        data = data.clone()
        labels = labels.clone()

        # Find indices of the target label
        target_indices = (labels == target_label).nonzero(as_tuple=True)[0]
        
        # Determine the number of labels to flip based on intensity
        num_to_flip = int(len(target_indices) * intensity)
        
        if num_to_flip > 0:
            # Randomly select indices to flip
            flip_indices = target_indices[torch.randperm(len(target_indices))[:num_to_flip]]
            
            # Flip the labels
            labels[flip_indices] = new_label
            
        poisoned_data.append(data)
        poisoned_labels.append(labels)

    # Create a new dataset and data loader with the poisoned data
    poisoned_dataset = TensorDataset(torch.cat(poisoned_data), torch.cat(poisoned_labels))
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=data_loader.batch_size, shuffle=True)
    
    return poisoned_loader

# In the future, other attacks can be added here
# def backdoor_injection(...)
# def gradient_manipulation(...)