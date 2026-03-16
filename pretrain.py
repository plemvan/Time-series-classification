import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # NOUVEAU: Pour le graphe
from torch.utils.data import DataLoader
from src.model import TimeSeriesEncoderCI, ForecastingModel
from src.dataset import InformerForecastingDataset
from src.preprocessing import scale_informer_data


def set_seed(seed=42):
    """Fixe toutes les seeds pour garantir la reproductibilité."""
    os.environ['PYTHONHASHSEED'] = str(seed) # Fixe les hashs de Python
    np.random.seed(seed)                   # Fixe le hasard de Numpy
    torch.manual_seed(seed)                # Fixe le hasard de PyTorch (CPU)
    torch.cuda.manual_seed(seed)           # Fixe le hasard de PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)       # Fixe le hasard multi-GPU (au cas où)
    
    # Force cuDNN à utiliser des algorithmes déterministes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_next_filename(base_name="graph", ext=".png"):
    i = 1
    while os.path.exists(f"{base_name}_{i}{ext}"):
        i += 1
    return f"{base_name}_{i}{ext}"

def main():
    set_seed(0)

    # --- HYPERPARAMÈTRES ---
    seq_len_past = 110      # Look-back window
    horizon = 10       # Prediction window
    hidden_dim = 32        # Must remain the same in finetune.py
    batch_size = 16
    epochs = 15
    learning_rate = 1e-4
    # -----------------------

    data_path = 'data/informer/ETTh1.csv'
    
    if os.path.exists(data_path):
        print("Loading data")
        df = pd.read_csv(data_path)
        raw_data = df.iloc[:, 1:].values 
    else:
        print("No data found. Randomly generated")
        raw_data = np.random.randn(10000, 7) 

    num_channels = raw_data.shape[1]
    print(f"Data shape: {raw_data.shape} ({num_channels} channels)")

    scaled_data, scaler = scale_informer_data(raw_data)

    # --- SPLIT CHRONOLOGIQUE (80% Train, 20% Val) ---
    train_size = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:]
    print(f"Split: Train={len(train_data)} points, Val={len(val_data)} points")

    print("Creating PyTorch Datasets and DataLoaders")
    train_dataset = InformerForecastingDataset(train_data, seq_len_past, horizon)
    val_dataset = InformerForecastingDataset(val_data, seq_len_past, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    encoder = TimeSeriesEncoderCI(hidden_dim=hidden_dim)
    forecasting_model = ForecastingModel(encoder, num_channels=num_channels, horizon=num_channels * horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecasting_model.to(device)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(forecasting_model.parameters(), lr=learning_rate)

    # Variables pour le Checkpointing et le Graphique
    best_val_loss = float('inf')
    best_encoder_path = "pretrained_encoder.pth"
    history_train_loss = []
    history_val_loss = []

    print("\nStarting pre-training phase (forecasting part)")
    
    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        forecasting_model.train()
        total_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = forecasting_model(batch_x)
            
            batch_y_flat = batch_y.reshape(batch_y.shape[0], -1)
            loss = criterion(predictions, batch_y_flat)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss) # Sauvegarde pour le graphe

        # --- VALIDATION PHASE ---
        forecasting_model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = forecasting_model(batch_x)
                batch_y_flat = batch_y.reshape(batch_y.shape[0], -1)
                loss = criterion(predictions, batch_y_flat)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss) # Sauvegarde pour le graphe

        print(f"    Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        # --- CHECKPOINTING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(encoder.state_dict(), best_encoder_path)

    # --- NOUVEAU : GÉNÉRATION DU GRAPHIQUE ---
    print("\nGenerating Pre-training Loss Curve...")
    os.makedirs("graphs", exist_ok=True) 
    
    plt.figure(figsize=(10, 7)) 
    
    plt.plot(range(1, epochs + 1), history_train_loss, label='Train MSE Loss', color='blue')
    plt.plot(range(1, epochs + 1), history_val_loss, label='Validation MSE Loss', color='orange')
    
    params_text = (
        f"Params: hidden_dim={hidden_dim} | batch_size={batch_size} | epochs={epochs}\n"
        f"seq_len={seq_len_past} | horizon={horizon} | lr={learning_rate}"
    )
    
    plt.title(params_text, fontsize=10, color='dimgray', pad=15)
    plt.suptitle('Pre-training (Forecasting) Loss over Epochs', fontsize=14, fontweight='bold', y=1.02)
    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout() 
    
    # On sauvegarde avec un préfixe différent pour ne pas mélanger avec la classification
    save_filename = get_next_filename(base_name="graphs/pretrain_graph", ext=".png")
    
    plt.savefig(save_filename, dpi=300, bbox_inches='tight') 
    print(f"Curve saved dynamically as '{save_filename}'")
    print("\nPre-training completed.")

if __name__ == "__main__":
    main()