import torch
import torch.nn as nn

class CNN_LSTM_IDS(nn.Module):
    """
    A simplified and effective CNN-LSTM model for Intrusion Detection.
    The CNN layer acts as a feature extractor on the input vector.
    The LSTM layer adds a recurrent component to the model.
    """
    def __init__(self, input_dim, output_dim=2):
        super(CNN_LSTM_IDS, self).__init__()
        # CNN layer. It expects input of shape (N, C_in, L_in).
        # We will treat our feature vector as L_in, and we have 1 channel (C_in).
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # The output from the CNN will be of shape (N, 64, input_dim/2).
        # This will be the input to the LSTM.
        # LSTM input shape: (N, L, H_in) where L is sequence length.
        # We will treat the 64 channels from the CNN as the features (H_in)
        # and the reduced feature dimension (input_dim/2) as the sequence length (L).
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # Input x shape: (batch_size, 1, input_dim)
        
        # Pass through CNN layer
        cnn_out = self.cnn(x) # Shape: (batch_size, 64, input_dim/2)
        
        # Permute the dimensions to be (batch_size, sequence_length, features) for the LSTM
        lstm_in = cnn_out.permute(0, 2, 1) # Shape: (batch_size, input_dim/2, 64)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(lstm_in)
        
        # We take the output of the last hidden state to pass to the fully connected layer
        out = self.fc(hn.squeeze(0))
        return out