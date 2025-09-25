import torch
import numpy as np
from collections import OrderedDict
from ids_model import CNN_LSTM_IDS

class DefenseSystem:
    def __init__(self, client_ids, honeypot_data, device, model_input_dim):
        self.fingerprints = {cid: [] for cid in client_ids}
        self.trust_scores = {cid: 1.0 for cid in client_ids}
        self.honeypot_data = honeypot_data
        self.device = device
        self.model_input_dim = model_input_dim

    # _calculate_fingerprint and update_trust_score_fingerprinting remain unchanged.
    def _calculate_fingerprint(self, model_update):
        flat_update = torch.cat([p.view(-1) for p in model_update.values()])
        return torch.linalg.norm(flat_update).item()

    def update_trust_score_fingerprinting(self, client_id, model_update):
        new_fingerprint = self._calculate_fingerprint(model_update)
        history = self.fingerprints[client_id]
        if len(history) > 2:
            mean_fp, std_fp = np.mean(history), np.std(history)
            if std_fp > 0 and abs(new_fingerprint - mean_fp) > 2 * std_fp:
                self.trust_scores[client_id] = max(0.1, self.trust_scores[client_id] * 0.9)
            else:
                self.trust_scores[client_id] = min(1.0, self.trust_scores[client_id] * 1.05)
        history.append(new_fingerprint)

    def validate_with_honeypot(self, client_id, client_model_weights, baseline_honeypot_acc):
        """
        Tests a client's model against the honeypot data using an adaptive baseline.
        """
        temp_model = CNN_LSTM_IDS(input_dim=self.model_input_dim).to(self.device)
        temp_model.load_state_dict(client_model_weights)
        temp_model.eval()
        
        data, target = self.honeypot_data
        data, target = data.to(self.device), target.to(self.device)
        
        with torch.no_grad():
            output = temp_model(data)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        
        client_honeypot_acc = correct / len(target)
        
        # --- ADAPTIVE TRUST LOGIC ---
        # Punish if the client's update significantly degrades performance
        # compared to the global model's baseline performance on the honeypot.
        if client_honeypot_acc < (baseline_honeypot_acc - 0.1): # e.g., drops by >10%
            self.trust_scores[client_id] = max(0.1, self.trust_scores[client_id] * 0.5)
        # A label-flipping attacker should cause a catastrophic drop, easily caught here.

    # robust_aggregation remains unchanged.
    def robust_aggregation(self, client_updates_with_ids):
        agg_weights = OrderedDict()
        first_client_id, first_update = client_updates_with_ids[0]
        for key in first_update.keys():
            agg_weights[key] = torch.zeros_like(first_update[key])
            
        total_trust_score = sum(self.trust_scores[cid] for cid, _ in client_updates_with_ids)
        
        if total_trust_score > 0:
            for client_id, weights in client_updates_with_ids:
                trust = self.trust_scores[client_id]
                for key in weights.keys():
                    agg_weights[key] += weights[key] * trust
            for key in agg_weights.keys():
                agg_weights[key] = torch.div(agg_weights[key], total_trust_score)
        else:
             # Fallback: if all trust is zero, perform a standard FedAvg
            num_clients = len(client_updates_with_ids)
            for _, weights in client_updates_with_ids:
                 for key in weights.keys():
                    agg_weights[key] += weights[key]
            for key in agg_weights.keys():
                agg_weights[key] = torch.div(agg_weights[key], num_clients)

        return agg_weights