import torch
import torch.nn as nn

class TimeSeriesEncoderCI(nn.Module):
    """Channel-Independent Encoder"""
    def __init__(self, hidden_dim): 
        super().__init__()
        # The network takes exactly 1 input channel, regardless of the dataset.
        self.conv_block = nn.Sequential(
            # Bloc 1
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_len → seq_len / 2
            nn.Dropout(p=0.1),

            # Bloc 2
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_len / 2 → seq_len / 4
            nn.Dropout(p=0.1),

            # Bloc 3
            nn.Conv1d(hidden_dim*2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # seq_len / 2 → seq_len / 4
            nn.Dropout(p=0.1),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # The size of the summary vector for each channel
        self.output_dim_per_channel = hidden_dim * 4 * 2

    def forward(self, x):
        batch_size, num_channels, seq_len = x.shape
        x_ci = x.reshape(batch_size * num_channels, 1, seq_len)
        
        # 1. Extraction des features convolutives
        features = self.conv_block(x_ci) 
        # Shape actuel : (batch * channels, hidden_dim * 4, seq_len_reduit)
        
        # 2. Dual Pooling en parallèle
        ap = self.avg_pool(features) # Shape: (batch * channels, hidden_dim * 4, 2)
        mp = self.max_pool(features) # Shape: (batch * channels, hidden_dim * 4, 2)
        
        # 3. Concaténation sur l'axe des filtres (dim=1)
        # Shape: (batch * channels, hidden_dim * 8, 2)
        combined = torch.cat([ap, mp], dim=1)
        
        # 4. On aplatit TOUT le reste (filtres + temps) pour chaque signal individuel
        # Shape: (batch * channels, hidden_dim * 8 * 2)
        combined = combined.flatten(start_dim=1)
        
        # 5. On remet sous la forme (Batch, Channels, Features_par_channel)
        combined = combined.reshape(batch_size, num_channels, self.output_dim_per_channel)
        
        # 6. Flatten final pour envoyer à la Head
        # Final output: (batch_size, num_channels * output_dim_per_channel)
        return combined.flatten(start_dim=1)



class ForecastingModel(nn.Module):
    def __init__(self, encoder, num_channels, horizon):
        super().__init__()
        self.encoder = encoder
        in_features = num_channels * encoder.output_dim_per_channel
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Léger, c'est de la régression
            nn.Linear(in_features // 2, horizon)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class ClassificationModel(nn.Module):
    """Classification Head (for LSST)"""
    def __init__(self, encoder, num_channels, num_classes):
        super().__init__()
        self.encoder = encoder
        in_features = num_channels * encoder.output_dim_per_channel
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)