import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_and_preprocess_nsl_kdd(num_clients):
    """
    Loads and preprocesses the NSL-KDD dataset.

    This function performs the following steps:
    1. Loads the KDDTrain+ and KDDTest+ files.
    2. Applies one-hot encoding to categorical features.
    3. Normalizes numerical features using MinMaxScaler.
    4. Splits the training data among the specified number of clients.
    5. Separates the test data into a main evaluation set and a smaller honeypot set
       for the server's defense mechanisms.

    Args:
        num_clients (int): The number of clients to distribute the training data among.

    Returns:
        tuple: A tuple containing:
            - client_data_loaders (list): A list of DataLoader objects for each client.
            - test_loader (DataLoader): A DataLoader for the main test set.
            - honeypot_data (tuple): A tuple of (data_tensor, label_tensor) for the honeypot.
            - input_dim (int): The number of features after preprocessing.
        Returns (None, None, None, None) if the dataset files are not found.
    """
    # Define column names for the dataset as they are not included in the files
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

    # Load training and testing data from local files
    try:
        df_train = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
        df_test = pd.read_csv('KDDTest+.txt', header=None, names=columns)
    except FileNotFoundError:
        print("Error: KDDTrain+.txt or KDDTest+.txt not found.")
        print("Please download the files from https://www.unb.ca/cic/datasets/nsl-kdd.html")
        return None, None, None, None

    # Combine train and test sets for consistent preprocessing
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Convert the multi-class labels to a binary classification problem: 'normal' vs 'attack'
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop('difficulty', axis=1) # Drop the unneeded difficulty column

    # One-hot encode all categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Normalize all numerical features to a range of [0, 1]
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('label')
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Separate the preprocessed data back into training and testing sets
    train_len = len(df_train)
    df_train = df.iloc[:train_len]
    df_test = df.iloc[train_len:]

    # Convert DataFrames to NumPy arrays with a specific float type for PyTorch compatibility
    X_train = df_train.drop('label', axis=1).values.astype(np.float32)
    y_train = df_train['label'].values
    X_test = df_test.drop('label', axis=1).values.astype(np.float32)
    y_test = df_test['label'].values

    # Reshape data to the 3D format required by the CNN-LSTM model: (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # --- Create Honeypot and Test Sets for the Server ---
    full_test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    
    # Split the full test set into a main test set and a smaller honeypot set
    test_size = len(full_test_dataset)
    honeypot_size = int(test_size * 0.20) # 20% for honeypots
    main_test_size = test_size - honeypot_size
    
    main_test_dataset, honeypot_dataset = random_split(full_test_dataset, [main_test_size, honeypot_size])

    test_loader = DataLoader(main_test_dataset, batch_size=128, shuffle=False)
    
    # Create the honeypot data as a single batch (data, labels) for easy access
    honeypot_loader = DataLoader(honeypot_dataset, batch_size=len(honeypot_dataset))
    honeypot_data = next(iter(honeypot_loader))

    # --- Distribute Training Data Among Clients ---
    client_data_loaders = []
    # Split the training data indices into non-overlapping chunks for each client
    data_indices = np.arange(len(X_train))
    client_indices_split = np.array_split(data_indices, num_clients)
    
    for i in range(num_clients):
        client_indices = client_indices_split[i]
        X_client = X_train[client_indices]
        y_client = y_train[client_indices]
        
        client_dataset = TensorDataset(torch.from_numpy(X_client), torch.from_numpy(y_client).long())
        client_loader = DataLoader(client_dataset, batch_size=64, shuffle=True)
        client_data_loaders.append(client_loader)
        
    # Get the input dimension (number of features) for the model
    input_dim = X_train.shape[2]
    
    return client_data_loaders, test_loader, honeypot_data, input_dim