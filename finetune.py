import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score, classification_report

from src.model import TimeSeriesEncoderCI, ClassificationModel
from src.dataset import LSSTClassificationDataset
from src.preprocessing import scale_lsst_data, encode_lsst_labels

def main():
    hidden_dim = 64
    batch_size = 16
    epochs = 100
    learning_rate = 1e-4

    ds = UCR_UEA_datasets()
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = ds.load_dataset("LSST")

    print("Preprocessing")
    X_train_scaled, X_test_scaled = scale_lsst_data(X_train_raw, X_test_raw)
    y_train_encoded, y_test_encoded, label_encoder = encode_lsst_labels(y_train_raw, y_test_raw)

    num_channels = X_train_scaled.shape[2] # LSST has 6 channels
    num_classes = len(np.unique(y_train_encoded)) # 14 classes

    print("Creating PyTorch Datasets and DataLoaders")
    train_dataset = LSSTClassificationDataset(X_train_scaled, y_train_encoded)
    test_dataset = LSSTClassificationDataset(X_test_scaled, y_test_encoded)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    encoder = TimeSeriesEncoderCI(hidden_dim=hidden_dim)
    
    # Load the pre-trained weights
    model_path = "pretrained_encoder.pth"
    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path))
        print("Pre-trained encoder weights loaded")
    else:
        print("'pretrained_encoder.pth' not found. Training from scratch.")

    classification_model = ClassificationModel(encoder, num_channels=num_channels, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_model.to(device)

    criterion = nn.CrossEntropyLoss() # Standard loss for multi-class classification
    optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)


    print("Starting fine-tuning phase (classification)")
    
    for epoch in range(epochs):
        classification_model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = classification_model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"    Epoch [{epoch+1}/{epochs}] | Train CrossEntropy Loss: {avg_loss:.4f}")

    
    print("\Evaluating the fine-tuned model on the Test Set")
    classification_model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad(): # No need to compute gradients during evaluation
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            predictions = classification_model(batch_x)
            
            # Get the index of the max log-probability (the predicted class)
            _, predicted_classes = torch.max(predictions, dim=1)
            
            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    target_names = [str(c) for c in label_encoder.classes_]

    print("="*50)
    print(f"Fine-Tuned Model Accuracy : {accuracy:.4f}")
    print("="*50)
    print("\nDetailed classification report:")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()