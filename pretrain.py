import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from src.model import TimeSeriesEncoderCI, ForecastingModel
from src.dataset import InformerForecastingDataset
from src.preprocessing import scale_informer_data

def main():
    seq_len_past = 96      # Look-back window (e.g., 96 hours)
    horizon = 24           # Prediction window (e.g., 24 hours)
    hidden_dim = 64        # Must remain the same in finetune.py
    batch_size = 32
    epochs = 5           
    learning_rate = 1e-3

    data_path = 'data/informer/ETTh1.csv'
    
    if os.path.exists(data_path):
        print("Loading data")
        df = pd.read_csv(data_path)
        raw_data = df.iloc[:, 1:].values 
    else:
        raw_data = np.random.randn(10000, 7) # 7 variables like ETTh1

    num_channels = raw_data.shape[1]
    print(f"Data shape: {raw_data.shape} ({num_channels} channels)")

    scaled_data, scaler = scale_informer_data(raw_data)

    print("Creating PyTorch Dataset and DataLoader")
    train_dataset = InformerForecastingDataset(scaled_data, seq_len_past, horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    encoder = TimeSeriesEncoderCI(hidden_dim=hidden_dim)
    forecasting_model = ForecastingModel(encoder, num_channels=num_channels, horizon=num_channels * horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecasting_model.to(device)

    criterion = nn.MSELoss() # Mean Squared Error for regression/forecasting
    optimizer = optim.Adam(forecasting_model.parameters(), lr=learning_rate)


    print("Starting pre-training phase (forecasting part)")
    forecasting_model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = forecasting_model(batch_x)
            
            # Flatten batch_y to match the head's output shape
            batch_y_flat = batch_y.reshape(batch_y.shape[0], -1)
            
            loss = criterion(predictions, batch_y_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"    Epoch [{epoch+1}/{epochs}] | Train MSE Loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), "pretrained_encoder.pth")
    print("Saved as 'pretrained_encoder.pth'")

if __name__ == "__main__":
    main()