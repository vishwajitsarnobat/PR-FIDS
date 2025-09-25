import torch
from collections import OrderedDict
from defenses import DefenseSystem
from ids_model import CNN_LSTM_IDS # Import the model to calculate baseline

class FLServer:
    def __init__(self, global_model, test_loader, device, client_ids, honeypot_data, model_input_dim):
        self.global_model = global_model
        self.test_loader = test_loader
        self.device = device
        self.honeypot_data = honeypot_data
        self.model_input_dim = model_input_dim
        self.defense_system = DefenseSystem(client_ids, honeypot_data, device, model_input_dim)

    def get_baseline_honeypot_accuracy(self):
        """Tests the current global model on the honeypot data."""
        self.global_model.eval()
        data, target = self.honeypot_data
        data, target = data.to(self.device), target.to(self.device)
        with torch.no_grad():
            output = self.global_model(data)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        return correct / len(target)

    def update_global_model(self, new_weights):
        self.global_model.load_state_dict(new_weights)

    def evaluate(self):
        self.global_model.to(self.device)
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total