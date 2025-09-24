import torch
from data_loader import load_and_preprocess_nsl_kdd
from ids_model import CNN_LSTM_IDS
from fl_client import FLClient
from fl_server import FLServer
import time

def main():
    # --- Configuration ---
    NUM_CLIENTS = 10
    NUM_ROUNDS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load and distribute data ---
    print("Loading and preprocessing NSL-KDD dataset...")
    client_loaders, test_loader, input_dim = load_and_preprocess_nsl_kdd(num_clients=NUM_CLIENTS)
    
    if client_loaders is None:
        return # Stop if data loading failed

    # --- Step 2: Initialize server and clients ---
    global_model = CNN_LSTM_IDS(input_dim=input_dim).to(device)
    server = FLServer(global_model, test_loader, device)
    
    clients = []
    for i in range(NUM_CLIENTS):
        client_model = CNN_LSTM_IDS(input_dim=input_dim)
        client = FLClient(client_id=i, model=client_model, data_loader=client_loaders[i], device=device)
        clients.append(client)
        
    print(f"\nInitialized {NUM_CLIENTS} clients and 1 server.")
    
    # --- Step 3: Run Federated Learning Process ---
    print("\n--- Starting Baseline Federated Learning (FedAvg) ---")
    for round_num in range(1, NUM_ROUNDS + 1):
        start_time = time.time()
        
        client_updates = []
        global_weights = server.global_model.state_dict()
        
        # Client training phase
        for client in clients:
            client.set_model_weights(global_weights)
            trained_weights = client.train(epochs=1)
            client_updates.append(trained_weights)
            
        # Server aggregation phase
        aggregated_weights = server.aggregate_updates(client_updates)
        server.update_global_model(aggregated_weights)
        
        # Evaluation phase
        accuracy = server.evaluate()
        
        round_time = time.time() - start_time
        print(f"Round {round_num}/{NUM_ROUNDS} | Accuracy: {accuracy:.2f}% | Time: {round_time:.2f}s")

if __name__ == '__main__':
    main()