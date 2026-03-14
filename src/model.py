import torch
import torch.nn as nn

class TimeSeriesEncoderCI(nn.Module):
    """Channel-Independent Encoder"""
    def __init__(self, hidden_dim): 
        super().__init__()
        # The network takes exactly 1 input channel, regardless of the dataset.
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global pooling along the time axis
        )
        # The size of the summary vector for each channel
        self.output_dim_per_channel = hidden_dim * 2

    def forward(self, x):
        # Initial x shape: (batch_size, num_channels, seq_len)
        batch_size, num_channels, seq_len = x.shape
        
        # Merge Batch and Channels
        # x_ci shape: (batch_size * num_channels, 1, seq_len)
        x_ci = x.reshape(batch_size * num_channels, 1, seq_len)
        
        # Pass through the network (which processes signals one by one)
        features = self.conv_block(x_ci) # Output: (batch_size * num_channels, output_dim_per_channel, 1)
        features = features.squeeze(-1)  # Output: (batch_size * num_channels, output_dim_per_channel)
        
        # Reshape the tensor to separate examples and channels
        features = features.reshape(batch_size, num_channels, self.output_dim_per_channel)
        
        # Flatten all channels together to send to the final head
        # Final output: (batch_size, num_channels * output_dim_per_channel)
        return features.flatten(start_dim=1)


class ForecastingModel(nn.Module):
    """Forecasting Head (for Informer)"""
    def __init__(self, encoder, num_channels, horizon):
        super().__init__()
        self.encoder = encoder
        # The head needs to know how many channels were processed to compute the input size
        in_features = num_channels * encoder.output_dim_per_channel
        self.head = nn.Linear(in_features, horizon)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class ClassificationModel(nn.Module):
    """Classification Head (for LSST)"""
    def __init__(self, encoder, num_channels, num_classes):
        super().__init__()
        self.encoder = encoder
        in_features = num_channels * encoder.output_dim_per_channel
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)