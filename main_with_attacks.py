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
    NUM_ROUNDS = 5
    MALICIOUS_CLIENT_PERCENTAGE = 0.3  # 30% of clients are malicious
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load and distribute data ---
    print("Loading and preprocessing NSL-KDD dataset...")
    client_loaders, test_loader, input_dim = load_and_preprocess_nsl_kdd(num_clients=NUM_CLIENTS)
    
    if client_loaders is None:
        return

    # --- Step 2: Initialize server and clients ---
    global_model = CNN_LSTM_IDS(input_dim=input_dim).to(device)
    server = FLServer(global_model, test_loader, device)
    
    # Designate malicious clients
    num_malicious_clients = int(NUM_CLIENTS * MALICIOUS_CLIENT_PERCENTAGE)
    malicious_client_ids = np.random.choice(range(NUM_CLIENTS), num_malicious_clients, replace=False)
    print(f"\nDesignating {num_malicious_clients} malicious clients: {malicious_client_ids}")

    clients = []
    for i in range(NUM_CLIENTS):
        is_malicious = i in malicious_client_ids
        client_model = CNN_LSTM_IDS(input_dim=input_dim)
        client = FLClient(
            client_id=i, 
            model=client_model, 
            data_loader=client_loaders[i], 
            device=device,
            is_malicious=is_malicious,
            attack_type='label_flipping' if is_malicious else None,
            attack_intensity=1.0 # Malicious clients flip 100% of their attack labels
        )
        clients.append(client)
        
    print(f"Initialized {NUM_CLIENTS} clients ({num_malicious_clients} malicious) and 1 server.")
    
    # --- Step 3: Run Federated Learning Process ---
    print("\n--- Starting FL with Label Flipping Attacks ---")
    for round_num in range(1, NUM_ROUNDS + 1):
        start_time = time.time()
        
        client_updates = []
        global_weights = server.global_model.state_dict()
        
        # Client training phase
        for client in clients:
            client.set_model_weights(global_weights)
            trained_weights = client.train(epochs=1)
            client_updates.append(trained_weights)
            
        # Server aggregation phase (still using basic FedAvg)
        aggregated_weights = server.aggregate_updates(client_updates)
        server.update_global_model(aggregated_weights)
        
        # Evaluation phase
        accuracy = server.evaluate()
        
        round_time = time.time() - start_time
        print(f"Round {round_num}/{NUM_ROUNDS} | Accuracy: {accuracy:.2f}% | Time: {round_time:.2f}s")

if __name__ == '__main__':
    main()