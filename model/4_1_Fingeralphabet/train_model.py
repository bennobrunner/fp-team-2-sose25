import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# Pfade für In- und Outputs
IN_CSV = "landmarks.csv"
OUT_MODEL = "asl_xgb.joblib"
OUT_LABELMAP_JOBLIB = "label_map.joblib"

# CSV einlesen und Label-Map bauen
def load_xy_and_labelmap(csv_path):
    df = pd.read_csv(csv_path)
    X = df.filter(like="f_").to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=int)
    label_map = dict(zip(df["y"], df["label"]))
    return X, y, label_map

# Split, Pipeline, Training, Evaluierung, Speichern
def main():
    X, y, label_map = load_xy_and_labelmap(IN_CSV)

    # Train/Val/Test-Split (70/15/15)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=42, stratify=y_tmp
    )

    # Modell-Pipeline
    pipe = Pipeline([
        ("clf", XGBClassifier(
            # mehr Bäume, damit Early Stopping „greifen“ kann
            n_estimators=4000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="gpu_hist",
            random_state=42,
        ))
    ])

    # Training + Vorhersage (mit Validation für Early Stopping)
    # Early Stopping wird über Fit-Parameter aktiviert
    pipe.fit(
        X_train, y_train,
        **{
            "clf__eval_set": [(X_val, y_val)],
            "clf__early_stopping_rounds": 100,
            "clf__verbose": False
        }
    )

    # Optional: Infos zur besten Iteration ausgeben
    xgb = pipe.named_steps["clf"]
    best_iter = getattr(xgb, "best_iteration", None)
    best_score = getattr(xgb, "best_score", None)
    if best_iter is not None:
        print(f"Beste Iteration (Early Stopping): {best_iter}")
    if best_score is not None:
        print(f"Bestes Val-mLOGLOSS: {best_score:.6f}")

    # Modellbewertung (Validation)
    y_val_pred = pipe.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    # Modellbewertung
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Konfusionsmatrix anlegen
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=90)
    plt.title("ASL Fingerspelling – Confusion Matrix (XGBoost GPU)")
    plt.tight_layout()
    plt.show()

    # Artefakte speichern
    joblib.dump(pipe, OUT_MODEL)
    joblib.dump(label_map, OUT_LABELMAP_JOBLIB)

# Skript starten
if __name__ == "__main__":
    main()