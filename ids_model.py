import torch
import torch.nn as nn

class CNN_LSTM_IDS(nn.Module):
    """
    CNN-LSTM model for Intrusion Detection.
    The CNN layer extracts spatial features, and the LSTM layer models temporal sequences.
    """
    def __init__(self, input_dim, output_dim=2):
        super(CNN_LSTM_IDS, self).__init__()
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1) 
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # The input x is expected to be (batch_size, sequence_length, num_features)
        # For our current data, sequence_length is 1.
        
        # Pass through LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # We only need the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(last_time_step_out)
        return out