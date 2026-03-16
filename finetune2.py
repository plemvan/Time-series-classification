import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model import TimeSeriesEncoderCI, ClassificationModel
from src.dataset import LSSTClassificationDataset
from src.preprocessing import scale_lsst_data, encode_lsst_labels


# Fonction pour trouver le prochain nom de fichier disponible
def get_next_filename(base_name="graph", ext=".png"):
    i = 1
    while os.path.exists(f"{base_name}_{i}{ext}"):
        i += 1
    return f"{base_name}_{i}{ext}"

def main():
    # --- Hyperparamètres ---
    hidden_dim = 64
    batch_size = 16
    epochs = 150
    freeze_epochs = 15    
    lr_head = 3e-4    
    lr_encoder_post_freeze = 1e-5
    lr_post_freeze = 5e-5
    # -----------------------

    ds = UCR_UEA_datasets()
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = ds.load_dataset("LSST")

    print("Preprocessing")
    X_train_scaled, X_test_scaled = scale_lsst_data(X_train_raw, X_test_raw)
    y_train_encoded, y_test_encoded, label_encoder = encode_lsst_labels(y_train_raw, y_test_raw)

    # Split Train / Validation (10%)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_encoded, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_train_encoded
    )

    num_channels = X_train_scaled.shape[2] 
    num_classes = len(np.unique(y_train_encoded)) 

    print(f"Dataset split: Train={len(X_train_split)}, Val={len(X_val_split)}, Test={len(X_test_scaled)}")

    train_dataset = LSSTClassificationDataset(X_train_split, y_train_split)
    val_dataset = LSSTClassificationDataset(X_val_split, y_val_split) 
    test_dataset = LSSTClassificationDataset(X_test_scaled, y_test_encoded)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    encoder = TimeSeriesEncoderCI(hidden_dim=hidden_dim)
    
    model_path = "pretrained_encoder.pth"
    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path))
        print("Pre-trained encoder weights loaded")
    else:
        print("'pretrained_encoder.pth' not found. Training from scratch.")

    classification_model = ClassificationModel(encoder, num_channels=num_channels, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model.to(device)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_split),
        y=y_train_split
    )
    smoothed_weights = np.sqrt(class_weights)
    weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    for param in classification_model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classification_model.parameters()),
        lr=lr_head
    )

    history_train_loss = []
    history_val_loss = []

    # --- Variables pour le Checkpointing ---
    best_val_loss = float('inf')
    best_model_path = "best_classification_model.pth"

    print("Starting fine-tuning phase (classification)")
    
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    for epoch in range(epochs):
        # 1. Gestion du Unfreezing
        if epoch == freeze_epochs:
            print(f"\n--- Epoch {epoch+1}: Unfreezing encoder with lr={lr_post_freeze} ---\n")
            for param in classification_model.encoder.parameters():
                param.requires_grad = True

            optimizer = optim.Adam([
                {'params': classification_model.encoder.parameters(), 'lr': lr_encoder_post_freeze},
                {'params': classification_model.head.parameters(),    'lr': lr_post_freeze}
            ])
            scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # 2. Train Phase
        classification_model.train()
        total_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = classification_model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # 3. Validation Phase
        classification_model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = classification_model(batch_x)
                loss = criterion(predictions, batch_y)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)

        print(f"    Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Sauvegarde du meilleur modèle ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classification_model.state_dict(), best_model_path)

    # --- Génération et sauvegarde dynamique du graphique ---
    print("\nGenerating Loss Curve...")
    os.makedirs("graphs", exist_ok=True) # Sécurité pour le dossier
    
    plt.figure(figsize=(10, 7)) 
    
    plt.plot(range(1, epochs + 1), history_train_loss, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), history_val_loss, label='Validation Loss', color='orange')
    
    if freeze_epochs < epochs:
        plt.axvline(x=freeze_epochs, color='red', linestyle='--', label='Unfreeze Encoder')
    
    params_text = (
        f"Params: hidden_dim={hidden_dim} | batch_size={batch_size} | epochs={epochs}\n"
        f"freeze_epochs={freeze_epochs} | lr_encoder={lr_post_freeze} | lr_head={lr_post_freeze}"
    )
    
    plt.title(params_text, fontsize=10, color='dimgray', pad=15)
    plt.suptitle('Training and Validation Loss over Epochs', fontsize=14, fontweight='bold', y=1.02)
    
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout() 
    
    save_filename = get_next_filename(base_name="graphs/graph", ext=".png")
    
    plt.savefig(save_filename, dpi=300, bbox_inches='tight') 
    print(f"Curve saved dynamically as '{save_filename}'")

    # --- Évaluation Finale sur le Test Set ---
    
    # --- Chargement du meilleur modèle pour l'inférence ---
    print(f"\nLoading the BEST model weights from '{best_model_path}' for evaluation...")
    classification_model.load_state_dict(torch.load(best_model_path))
    
    print("Evaluating the fine-tuned model on the Test Set")
    classification_model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            predictions = classification_model(batch_x)
            
            _, predicted_classes = torch.max(predictions, dim=1)
            
            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    target_names = [str(c) for c in label_encoder.classes_]

    print("="*50)
    print(f"Fine-Tuned Model Accuracy : {accuracy:.4f}")
    print("="*50)
    print("\nDetailed classification report:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()