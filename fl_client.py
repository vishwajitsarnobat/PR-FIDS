import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from attacks import label_flipping_attack # Import the attack function

class FLClient:
    """
    Represents a client in the Federated Learning system.
    Can be configured to be malicious.
    """
    def __init__(self, client_id, model, data_loader, device, is_malicious=False, attack_type=None, attack_intensity=1.0):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.attack_intensity = attack_intensity
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = CrossEntropyLoss()

    def set_model_weights(self, server_weights):
        """Load model weights from the server."""
        self.model.load_state_dict(server_weights)

    def train(self, epochs=1):
        """
        Train the model on local data.
        If the client is malicious, it applies a poisoning attack first.
        """
        training_loader = self.data_loader

        # If the client is malicious, poison the data before training
        if self.is_malicious and self.attack_type == 'label_flipping':
            # print(f"Client {self.client_id} is malicious and performing a label flipping attack.")
            training_loader = label_flipping_attack(
                self.data_loader, 
                intensity=self.attack_intensity
            )

        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            for data, target in training_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()