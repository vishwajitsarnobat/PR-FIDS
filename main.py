import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
from data_loader import load_and_preprocess_nsl_kdd
from ids_model import CNN_LSTM_IDS
from fl_client import FLClient
from fl_server import FLServer

# --- MASTER CONFIGURATION ---
SCENARIOS = {
    "baseline": {"attack_type": None, "defenses_on": False},
    "attack_label_flipping": {"attack_type": "label_flipping", "defenses_on": False},
    "attack_backdoor": {"attack_type": "backdoor", "defenses_on": False},
    "defended_label_flipping": {"attack_type": "label_flipping", "defenses_on": True},
    "defended_backdoor": {"attack_type": "backdoor", "defenses_on": True},
}

# --- SELECT SCENARIOS TO RUN ---
# You can run one or more scenarios at a time. The results will be saved together.
scenarios_to_run = [
    "baseline",
    "attack_label_flipping",
    "attack_backdoor",
    "defended_label_flipping",
    "defended_backdoor"
]

# --- SIMULATION PARAMETERS ---
NUM_CLIENTS = 10
NUM_ROUNDS = 10
MALICIOUS_CLIENT_PERCENTAGE = 0.3

def run_simulation(scenario_name, attack_type, defenses_on):
    """Runs a single federated learning simulation for a given scenario."""
    print(f"\n{'='*50}\nRunning Scenario: {scenario_name}\n{'='*50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    client_loaders, test_loader, honeypot_data, input_dim = load_and_preprocess_nsl_kdd(num_clients=NUM_CLIENTS)
    if client_loaders is None: return []

    global_model = CNN_LSTM_IDS(input_dim=input_dim).to(device)
    
    num_malicious_clients = int(NUM_CLIENTS * MALICIOUS_CLIENT_PERCENTAGE) if attack_type else 0
    malicious_client_ids = np.random.choice(range(NUM_CLIENTS), num_malicious_clients, replace=False).tolist()
    if attack_type:
        print(f"Designating {num_malicious_clients} malicious clients: {malicious_client_ids} with attack '{attack_type}'")

    all_client_ids = list(range(NUM_CLIENTS))
    server = FLServer(global_model, test_loader, device, all_client_ids, honeypot_data, input_dim)
    
    clients = []
    for i in range(NUM_CLIENTS):
        is_malicious = i in malicious_client_ids
        client_model = CNN_LSTM_IDS(input_dim=input_dim)
        client = FLClient(
            client_id=i, model=client_model, data_loader=client_loaders[i], device=device,
            is_malicious=is_malicious,
            attack_type=attack_type if is_malicious else None
        )
        clients.append(client)
        
    results = []
    for round_num in range(1, NUM_ROUNDS + 1):
        client_updates_with_ids = []
        global_weights = server.global_model.state_dict()
        
        for client in clients:
            client.set_model_weights(global_weights)
            trained_weights = client.train(epochs=1)
            client_updates_with_ids.append((client.client_id, trained_weights))
            
        if defenses_on:
            baseline_acc = server.get_baseline_honeypot_accuracy()
            for client_id, update in client_updates_with_ids:
                server.defense_system.validate_with_honeypot(client_id, update, baseline_acc)
                server.defense_system.update_trust_score_fingerprinting(client_id, update)
            aggregated_weights = server.defense_system.robust_aggregation(client_updates_with_ids)
        else: # Basic FedAvg
            updates = [update for _, update in client_updates_with_ids]
            agg_weights_ordered = OrderedDict()
            for key in global_weights.keys():
                agg_weights_ordered[key] = torch.stack([w[key] for w in updates]).mean(dim=0)
            aggregated_weights = agg_weights_ordered

        server.update_global_model(aggregated_weights)
        accuracy = server.evaluate()
        
        print(f"  Round {round_num}/{NUM_ROUNDS} | Accuracy: {accuracy:.2f}%")
        results.append({"scenario": scenario_name, "round": round_num, "accuracy": accuracy})

    return results

def main():
    """Main function to run selected scenarios and save results."""
    all_results = []
    for scenario_name in scenarios_to_run:
        if scenario_name in SCENARIOS:
            config = SCENARIOS[scenario_name]
            scenario_results = run_simulation(scenario_name, config["attack_type"], config["defenses_on"])
            all_results.extend(scenario_results)
        else:
            print(f"Warning: Scenario '{scenario_name}' not found in configuration.")

    # Save results to a CSV file for later analysis
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv("simulation_results.csv", index=False)
        print(f"\nSimulation complete. Results saved to simulation_results.csv")

if __name__ == '__main__':
    main()