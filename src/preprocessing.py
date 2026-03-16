import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_lsst_labels(y_train, y_test):
    """
    Encodes the original LSST class labels (e.g., 15, 42, 90) into contiguous integers from 0 to 13.
    """
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    return y_train_encoded, y_test_encoded, encoder

def scale_lsst_data(X_train, X_test):
    """
    Normalisation cohérente avec ETTh : StandardScaler par canal.
    X shape: (Samples, Time, Channels)
    """
    n_train, seq_len, n_channels = X_train.shape
    n_test = X_test.shape[0]

    # Reshape in 2D for StandardScaler
    X_train_2d = X_train.reshape(-1, n_channels)  # (Samples*Time, Channels)
    X_test_2d  = X_test.reshape(-1, n_channels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, seq_len, n_channels)
    X_test_scaled  = scaler.transform(X_test_2d).reshape(n_test, seq_len, n_channels)

    return X_train_scaled, X_test_scaled

def scale_informer_data(data):
    """
    Scales the 2D Informer dataset (Timestamps, Channels) for the forecasting pre-training phase.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler