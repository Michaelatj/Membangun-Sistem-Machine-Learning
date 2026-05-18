"""
modelling.py
-------------
Kriteria 2 - Basic
Melatih model klasifikasi PHQ_9_Severity menggunakan RandomForestClassifier
dengan MLflow autolog. Artefak disimpan secara lokal di MLflow Tracking UI.

Cara pakai:
    python modelling.py

Pastikan dataset preprocessing sudah ada:
    social_media_mental_health_preprocessing/train.csv
    social_media_mental_health_preprocessing/test.csv
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
TRAIN_PATH = "social_media_mental_health_preprocessing/train.csv"
TEST_PATH  = "social_media_mental_health_preprocessing/test.csv"
TARGET_COL = "PHQ_9_Severity"
MLFLOW_EXPERIMENT = "Social_Media_Mental_Health"

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_data():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]
    X_test  = df_test.drop(columns=[TARGET_COL])
    y_test  = df_test[TARGET_COL]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test = load_data()

    # Set MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Aktifkan autolog sklearn
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nTest Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\n[DONE] Run berhasil. Cek MLflow UI dengan: mlflow ui")

if __name__ == "__main__":
    main()
