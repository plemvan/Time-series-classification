import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

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
    Scales the 3D LSST time series (Samples, Time, Channels) to have zero mean and unit variance.
    """
    # TimeSeriesScalerMeanVariance is designed specifically for 3D time series
    scaler = TimeSeriesScalerMeanVariance()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def scale_informer_data(data):
    """
    Scales the 2D Informer dataset (Timestamps, Channels) for the forecasting pre-training phase.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler