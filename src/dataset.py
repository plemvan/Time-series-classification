import torch
from torch.utils.data import Dataset
import numpy as np

class InformerForecastingDataset(Dataset):
    """
    Dataset for phase 1: Pre-training (Forecasting).
    Takes a long multivariate time series and creates sliding windows.
    """
    def __init__(self, data, seq_len_past, horizon):
        # Expected data: numpy array of shape (Total_Timestamps, Channels)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len_past = seq_len_past
        self.horizon = horizon

    def __len__(self):
        # The number of windows that can be extracted
        return len(self.data) - self.seq_len_past - self.horizon + 1

    def __getitem__(self, idx):
        # Extract the past (X)
        x_past = self.data[idx : idx + self.seq_len_past]
        
        # Extract the future to predict (Y)
        y_future = self.data[idx + self.seq_len_past : idx + self.seq_len_past + self.horizon]
        
        # PyTorch Conv1d expects the (Channels, Time) format, so we transpose using permute
        x_past = x_past.permute(1, 0)
        y_future = y_future.permute(1, 0)
        
        return x_past, y_future


class LSSTClassificationDataset(Dataset):
    """
    Dataset for phase 2: Adaptation (Fine-tuning on LSST).
    """
    def __init__(self, X, y):
        # Expected X from tslearn: (Samples, Time, Channels)
        # Expected y: (Samples,) encoded from 0 to 13
        # Convert to tensors and transpose to get (Samples, Channels, Time)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]