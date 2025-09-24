import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

class FLClient:
    """
    Represents a client in the Federated Learning system.
    """
    def __init__(self, client_id, model, data_loader, device):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = CrossEntropyLoss()

    def set_model_weights(self, server_weights):
        """Load model weights from the server."""
        self.model.load_state_dict(server_weights)

    def train(self, epochs=1):
        """Train the model on local data."""
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()