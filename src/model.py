import torch
import torch.nn as nn

class TimeSeriesEncoderCI(nn.Module):
    """Channel-Independent Encoder"""
    def __init__(self, hidden_dim): 
        super().__init__()
        # The network takes exactly 1 input channel, regardless of the dataset.
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_len -> seq_len / 2
            nn.Dropout(p=0.1),

            # Block 2
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_len / 2 -> seq_len / 4
            nn.Dropout(p=0.1),

            # Block 3
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_len / 4 -> seq_len / 8
            nn.Dropout(p=0.1),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # The size of the summary vector for each channel
        self.output_dim_per_channel = hidden_dim * 4 * 2

    def forward(self, x):
        #  Reshape for Channel Independence
        batch_size, num_channels, seq_len = x.shape
        x_ci = x.reshape(batch_size * num_channels, 1, seq_len)

        features = self.conv_block(x_ci) 
        # Current shape: (batch * channels, hidden_dim * 4, reduced_seq_len)
        
        #  Parallel Dual Pooling
        ap = self.avg_pool(features) # Shape: (batch * channels, hidden_dim * 4, 1)
        mp = self.max_pool(features) # Shape: (batch * channels, hidden_dim * 4, 1)
        
        # Concatenation along the filter axis (dim=1)
        # Shape: (batch * channels, hidden_dim * 8, 1)
        combined = torch.cat([ap, mp], dim=1)
        
        # Flatten the remaining dimensions (filters + time) for each individual signal
        # Shape: (batch * channels, hidden_dim * 8)
        combined = combined.flatten(start_dim=1)
        
        # Reshape back to separate the batch and channel dimensions
        # Shape: (batch_size, num_channels, features_per_channel)
        combined = combined.reshape(batch_size, num_channels, self.output_dim_per_channel)
        
        # Final flatten to pass the features to the specific Head
        # Final output shape: (batch_size, num_channels * output_dim_per_channel)
        return combined.flatten(start_dim=1)


class ForecastingModel(nn.Module):
    """Forecasting Head (for ETTh1 Pre-training)"""
    def __init__(self, encoder, num_channels, horizon):
        super().__init__()
        self.encoder = encoder
        in_features = num_channels * encoder.output_dim_per_channel
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Light dropout, as this is a regression task
            nn.Linear(in_features // 2, horizon)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class ClassificationModel(nn.Module):
    """Classification Head (for LSST Fine-tuning)"""
    def __init__(self, encoder, num_channels, num_classes):
        super().__init__()
        self.encoder = encoder
        in_features = num_channels * encoder.output_dim_per_channel
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.4),  # Heavier dropout for classification to prevent overfitting
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)