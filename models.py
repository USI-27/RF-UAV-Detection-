import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for background noise anomaly detection.
    Inputs: Spectrograms of shape (batch, 5, 129, 9) - representing 5 UCA channels.
    Outputs: Reconstructed spectrograms of the same shape.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 5, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # input shape: (batch, 5, 129, 9)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class FeatureCNN(nn.Module):
    """
    2D CNN with final classification layers removed.
    Inputs: Spectrograms of shape (batch, 5, 256, 9).
    Outputs: 1D Feature Embeddings of shape (batch, embedding_dim).
    """
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Shape: (batch, 16, 128, 4)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # Shape: (batch, 32, 64, 2)
        )
        self.fc = nn.Linear(32 * 64 * 2, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class TemporalBiLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal verification of FHSS hopping patterns.
    Inputs: Sequences of feature vectors of shape (seq_len=5, batch, embedding_dim=64).
    Outputs: Confidence score between 0.0 and 1.0.
    """
    def __init__(self, embedding_dim=64, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=False,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, seq):
        # seq shape: (seq_len, batch, embedding_dim)
        lstm_out, _ = self.lstm(seq)
        # Take the output of the final time step
        last_step_out = lstm_out[-1] # shape: (batch, hidden_dim * 2)
        confidence = self.fc(last_step_out) # shape: (batch, 1)
        return confidence
