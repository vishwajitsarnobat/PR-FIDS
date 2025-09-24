import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess_nsl_kdd(num_clients):
    """
    Loads, preprocesses the NSL-KDD dataset, and distributes it among clients.
    """
    # Define column names for the dataset
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    # Load training and testing data
    # NOTE: You must download the 'KDDTrain+.txt' and 'KDDTest+.txt' files and place them
    # in the same directory as this script.
    try:
        df_train = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
        df_test = pd.read_csv('KDDTest+.txt', header=None, names=columns)
    except FileNotFoundError:
        print("Error: KDDTrain+.txt or KDDTest+.txt not found.")
        print("Please download them from https://www.unb.ca/cic/datasets/nsl-kdd.html")
        return None, None, None

    # Combine for preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Label mapping: 'normal' vs 'attack'
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop('difficulty', axis=1)

    # One-hot encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Normalize numerical features
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('label')
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Separate train and test data again
    train_len = len(df_train)
    df_train = df.iloc[:train_len]
    df_test = df.iloc[train_len:]

    X_train = df_train.drop('label', axis=1).values
    y_train = df_train['label'].values
    X_test = df_test.drop('label', axis=1).values
    y_test = df_test['label'].values

    # Reshape data for CNN-LSTM (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Create a global test set for the server
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Distribute training data among clients
    client_data_loaders = []
    # Split training data into non-overlapping chunks for each client
    data_split = np.array_split(np.arange(len(X_train)), num_clients)
    for i in range(num_clients):
        client_indices = data_split[i]
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]
        client_dataset = TensorDataset(torch.from_numpy(X_client).float(), torch.from_numpy(y_client).long())
        client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        client_data_loaders.append(client_loader)
        
    input_dim = X_train.shape[2]
    
    return client_data_loaders, test_loader, input_dim