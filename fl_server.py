import torch
from collections import OrderedDict

class FLServer:
    """
    The Federated Learning Server that orchestrates the training process.
    """
    def __init__(self, global_model, test_loader, device):
        self.global_model = global_model
        self.test_loader = test_loader
        self.device = device

    def aggregate_updates(self, client_updates):
        """
        Aggregate model updates from clients using Federated Averaging (FedAvg).
        """
        agg_weights = OrderedDict()
        num_clients = len(client_updates)

        # Initialize aggregated weights with zeros
        for key in self.global_model.state_dict().keys():
            agg_weights[key] = torch.zeros_like(self.global_model.state_dict()[key])

        # Sum up the weights from all clients
        for weights in client_updates:
            for key in weights.keys():
                agg_weights[key] += weights[key]

        # Average the weights
        for key in agg_weights.keys():
            agg_weights[key] = torch.div(agg_weights[key], num_clients)

        return agg_weights

    def update_global_model(self, new_weights):
        """Update the global model with aggregated weights."""
        self.global_model.load_state_dict(new_weights)

    def evaluate(self):
        """Evaluate the global model on the test dataset."""
        self.global_model.to(self.device)
        self.global_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy