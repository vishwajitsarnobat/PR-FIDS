import torch
import numpy as np
from data_loader import load_and_preprocess_nsl_kdd
from ids_model import CNN_LSTM_IDS
from fl_client import FLClient
from fl_server import FLServer
import time

def main():
    # --- Configuration ---
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10
    MALICIOUS_CLIENT_PERCENTAGE = 0.3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    client_loaders, test_loader, honeypot_data, input_dim = load_and_preprocess_nsl_kdd(num_clients=NUM_CLIENTS)
    if client_loaders is None: return

    # --- Initialize server and clients ---
    global_model = CNN_LSTM_IDS(input_dim=input_dim).to(device)
    
    num_malicious_clients = int(NUM_CLIENTS * MALICIOUS_CLIENT_PERCENTAGE)
    malicious_client_ids = np.random.choice(range(NUM_CLIENTS), num_malicious_clients, replace=False).tolist()
    print(f"\nDesignating {num_malicious_clients} malicious clients: {malicious_client_ids}")

    all_client_ids = list(range(NUM_CLIENTS))
    server = FLServer(global_model, test_loader, device, all_client_ids, honeypot_data, input_dim)
    
    clients = []
    for i in range(NUM_CLIENTS):
        is_malicious = i in malicious_client_ids
        client_model = CNN_LSTM_IDS(input_dim=input_dim)
        client = FLClient(
            client_id=i, model=client_model, data_loader=client_loaders[i], device=device,
            is_malicious=is_malicious,
            attack_type='label_flipping' if is_malicious else None,
            attack_intensity=1.0
        )
        clients.append(client)
        
    print(f"Initialized {NUM_CLIENTS} clients. Applying FINAL Adaptive 3-Layer Defense.")
    
    # --- Run Federated Learning Process ---
    print("\n--- Starting FL with Adaptive 3-Layer Defense System ---")
    for round_num in range(1, NUM_ROUNDS + 1):
        start_time = time.time()
        
        client_updates_with_ids = []
        global_weights = server.global_model.state_dict()
        
        for client in clients:
            client.set_model_weights(global_weights)
            trained_weights = client.train(epochs=1)
            client_updates_with_ids.append((client.client_id, trained_weights))
            
        # --- ADAPTIVE DEFENSE SYSTEM IN ACTION ---
        
        # Server calculates baseline performance on the trusted data
        baseline_acc = server.get_baseline_honeypot_accuracy()

        for client_id, update in client_updates_with_ids:
            # Layer 2: Active validation using the adaptive baseline
            server.defense_system.validate_with_honeypot(client_id, update, baseline_acc)
            # Layer 1: Passive validation (still useful as a secondary signal)
            server.defense_system.update_trust_score_fingerprinting(client_id, update)

        # Layer 3: Robust aggregation using correctly updated trust scores
        aggregated_weights = server.defense_system.robust_aggregation(client_updates_with_ids)
        server.update_global_model(aggregated_weights)
        
        accuracy = server.evaluate()
        
        scores = server.defense_system.trust_scores
        formatted_scores = [f"C{cid}: {score:.2f}" for cid, score in scores.items()]
        
        round_time = time.time() - start_time
        print(f"Round {round_num}/{NUM_ROUNDS} | Accuracy: {accuracy:.2f}% | Baseline Honeypot Acc: {baseline_acc:.2f}")
        print(f"  Trust Scores: {', '.join(formatted_scores)}")
        print("-" * 20)

if __name__ == '__main__':
    main()