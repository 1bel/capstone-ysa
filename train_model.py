"""
train_model.py
--------------
Trains the SVM (RBF kernel) binary classifier for the canine active
cooling system and saves the fitted pipeline to disk with joblib.

Usage
-----
# Train using a real sensor-log CSV (must have columns:
# body_temp, respiratory_rate, ambient_temp, label[optional])
python train_model.py --data my_sensor_logs.csv --output cooling_svm_model.joblib

# Train using the built-in synthetic generator (no --data given)
python train_model.py --output cooling_svm_model.joblib

If your CSV does NOT already contain a 'label' column, this script will
automatically compute it from the 2-out-of-3 heat-stress rule defined in
data_utils.py.
"""

import argparse

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from data_utils import generate_synthetic_dataset, label_dataframe
from svm_model import FEATURE_COLUMNS, build_svm_pipeline


def load_dataset(data_path: str | None) -> pd.DataFrame:
    if data_path is None:
        print("[INFO] No --data path given. Generating synthetic dataset for prototyping...")
        return generate_synthetic_dataset()

    df = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if "label" not in df.columns:
        print("[INFO] No 'label' column found. Computing labels from the 2-of-3 rule...")
        df["label"] = label_dataframe(df)

    return df


def main():
    parser = argparse.ArgumentParser(description="Train SVM (RBF) cooling-trigger classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to sensor log CSV")
    parser.add_argument(
        "--output", type=str, default="cooling_svm_model.joblib", help="Path to save trained model"
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(df["label"].value_counts().rename({0: "OFF (0)", 1: "ON (1)"}))

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # --- Hyperparameter search for the RBF kernel -------------------------
    pipeline = build_svm_pipeline()
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.1, 1],
    }

    print("[INFO] Running GridSearchCV (5-fold) over C and gamma...")
    print("[INFO] This runs 20 parameter combos x 5 folds = 100 fits. "
          "Progress will print below — this can take a minute or two, it is not frozen.")
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=2
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"[INFO] Best params: {grid.best_params_}")
    print(f"[INFO] Best CV F1 score: {grid.best_score_:.4f}")

    # --- Evaluation on held-out test set -----------------------------------
    y_pred = best_model.predict(X_test)

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [0=OFF, 1=ON]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["OFF (0)", "ON (1)"]))

    # --- Train set check (compare with test to detect overfitting) ---------
    y_train_pred = best_model.predict(X_train)
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}  "
          f"| Test Accuracy: {accuracy_score(y_test, y_pred):.4f}  "
          f"(large gap = overfitting)")

    # --- Save model ----------------------------------------------------
    joblib.dump(best_model, args.output)
    print(f"[INFO] Saved trained model to: {args.output}")


if __name__ == "__main__":
    main()
