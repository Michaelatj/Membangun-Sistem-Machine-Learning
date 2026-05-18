"""
modelling_tuning.py
--------------------
Kriteria 2 - Skilled
Melatih model klasifikasi PHQ_9_Severity menggunakan RandomForestClassifier
dengan hyperparameter tuning (GridSearchCV) dan MANUAL LOGGING MLflow.

Manual logging mencakup metrik yang sama dengan autolog:
  - training_accuracy_score
  - training_f1_score (weighted)
  - training_precision_score (weighted)
  - training_recall_score (weighted)
  - test_accuracy_score
  - test_f1_score (weighted)
  - test_precision_score (weighted)
  - test_recall_score (weighted)
  - best hyperparameters

Cara pakai:
    python modelling_tuning.py

Pastikan dataset preprocessing sudah ada:
    social_media_mental_health_preprocessing/train.csv
    social_media_mental_health_preprocessing/test.csv
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
TRAIN_PATH = "social_media_mental_health_preprocessing/train.csv"
TEST_PATH  = "social_media_mental_health_preprocessing/test.csv"
TARGET_COL = "PHQ_9_Severity"
MLFLOW_EXPERIMENT = "Social_Media_Mental_Health"
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

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

    print(f"[load_data] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ──────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
def tune_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n[tune_model] Best params : {grid_search.best_params_}")
    print(f"[tune_model] Best CV acc  : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

# ──────────────────────────────────────────────
# BUAT ARTEFAK CONFUSION MATRIX
# ──────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, labels, path):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix - Best Model (Test Set)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[artifact] Confusion matrix saved to {path}")

# ──────────────────────────────────────────────
# BUAT ARTEFAK FEATURE IMPORTANCE
# ──────────────────────────────────────────────
def save_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color="steelblue", edgecolor="black")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importances - RandomForest Best Model")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[artifact] Feature importance saved to {path}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    X_train, X_test, y_train, y_test = load_data()

    # Tuning
    best_model, best_params, best_cv_acc = tune_model(X_train, y_train)

    # Prediksi
    y_train_pred = best_model.predict(X_train)
    y_test_pred  = best_model.predict(X_test)

    # ── METRIK TRAINING ──
    train_acc       = accuracy_score(y_train, y_train_pred)
    train_f1        = f1_score(y_train, y_train_pred, average="weighted")
    train_precision = precision_score(y_train, y_train_pred, average="weighted")
    train_recall    = recall_score(y_train, y_train_pred, average="weighted")

    # ── METRIK TEST ──
    test_acc        = accuracy_score(y_test, y_test_pred)
    test_f1         = f1_score(y_test, y_test_pred, average="weighted")
    test_precision  = precision_score(y_test, y_test_pred, average="weighted")
    test_recall     = recall_score(y_test, y_test_pred, average="weighted")

    # Cross-val score pada best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    print(f"\n{'='*50}")
    print(f"  RESULTS - Best Model")
    print(f"{'='*50}")
    print(f"  Train Accuracy   : {train_acc:.4f}")
    print(f"  Test  Accuracy   : {test_acc:.4f}")
    print(f"  Test  F1 (weighted): {test_f1:.4f}")
    print(f"  CV Mean Accuracy : {cv_mean:.4f} ± {cv_std:.4f}")
    print()
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # ── SIMPAN ARTEFAK LOKAL ──
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    cr_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")

    label_map = sorted(y_test.unique())
    save_confusion_matrix(y_test, y_test_pred, label_map, cm_path)
    save_feature_importance(best_model, list(X_train.columns), fi_path)

    with open(cr_path, "w") as f:
        f.write(classification_report(y_test, y_test_pred))
    print(f"[artifact] Classification report saved to {cr_path}")

    # ──────────────────────────────────────────
    # MLFLOW MANUAL LOGGING
    # ──────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="RandomForest_Tuning_ManualLog"):

        # Log hyperparameter terbaik
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "accuracy")

        # Log metrik training (sama dengan autolog)
        mlflow.log_metric("training_accuracy_score",   train_acc)
        mlflow.log_metric("training_f1_score",         train_f1)
        mlflow.log_metric("training_precision_score",  train_precision)
        mlflow.log_metric("training_recall_score",     train_recall)

        # Log metrik test (sama dengan autolog)
        mlflow.log_metric("test_accuracy_score",       test_acc)
        mlflow.log_metric("test_f1_score",             test_f1)
        mlflow.log_metric("test_precision_score",      test_precision)
        mlflow.log_metric("test_recall_score",         test_recall)

        # Log metrik tambahan (cross-validation)
        mlflow.log_metric("cv_mean_accuracy",          cv_mean)
        mlflow.log_metric("cv_std_accuracy",           cv_std)
        mlflow.log_metric("best_cv_accuracy",          best_cv_acc)

        # Log artefak
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(cr_path)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[MLflow] Run ID    : {run_id}")
        print(f"[MLflow] Experiment: {MLFLOW_EXPERIMENT}")
        print("\n[DONE] Run berhasil. Cek MLflow UI dengan: mlflow ui")

if __name__ == "__main__":
    main()
