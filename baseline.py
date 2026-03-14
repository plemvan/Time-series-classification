import numpy as np
from tslearn.datasets import UCR_UEA_datasets
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from aeon.transformations.collection.convolution_based import Rocket
from src.preprocessing import scale_lsst_data, encode_lsst_labels

def main():
    # Load Data
    print("Loading LSST dataset from tslearn")
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    # Preprocessing
    print("Preprocessing data")
    X_train_scaled, X_test_scaled = scale_lsst_data(X_train, X_test)
    y_train_encoded, y_test_encoded, label_encoder = encode_lsst_labels(y_train, y_test)

    # Format Conversion for ROCKET (aeon)
    # tslearn format: (n_samples, n_timestamps, n_channels)
    # aeon format:    (n_samples, n_channels, n_timestamps)
    X_train_aeon = np.transpose(X_train_scaled, (0, 2, 1))
    X_test_aeon = np.transpose(X_test_scaled, (0, 2, 1))

    # Feature Extraction with ROCKET
    print("Extracting features with ROCKET")
    rocket = Rocket(n_kernels=10000, random_state=42)
    X_train_features = rocket.fit_transform(X_train_aeon)
    X_test_features = rocket.transform(X_test_aeon)

    # Classifier Training
    print("Training the Ridge Classifier")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_features, y_train_encoded)

    # Evaluation
    print("Evaluating the baseline on the test set")
    y_pred = classifier.predict(X_test_features)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"ROCKET Baseline Accuracy : {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    target_names = [str(c) for c in label_encoder.classes_]
    print(classification_report(y_test_encoded, y_pred, target_names=target_names))

if __name__ == "__main__":
    main()