import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb

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
    num_classes = int(np.unique(y).size)

    # Train/Val/Test-Split (70/15/15)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=42, stratify=y_tmp
    )

    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test)

    # XGBoost-Parameter
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "tree_method": "hist",
        "device": "cuda",   # "cuda" für GPU, sonst "cpu"
        "seed": 42,
    }

    # Training + Vorhersage (mit Validation für Early Stopping)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # Beste Iteration ausgeben
    if hasattr(booster, "best_iteration"):
        print(f"Beste Iteration (Early Stopping): {booster.best_iteration}")
    if hasattr(booster, "best_score"):
        try:
            print(f"Bestes Val-mLOGLOSS: {float(booster.best_score):.6f}")
        except Exception:
            pass

    # Modellbewertung (Validation)
    y_val_pred = np.asarray(booster.predict(dval)).argmax(axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation: {val_acc:.4f}")
    print("Validation nach Klassen:")
    print(classification_report(y_val, y_val_pred))

    # Modellbewertung
    y_pred = np.asarray(booster.predict(dtest)).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Konfusionsmatrix anlegen
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=90)
    plt.title("ASL Fingerspelling – Confusion Matrix (XGBoost GPU)")
    plt.tight_layout()
    plt.show()

    # Artefakte speichern
    booster.save_model(OUT_MODEL)
    joblib.dump(label_map, OUT_LABELMAP_JOBLIB)

# Skript starten
if __name__ == "__main__":
    main()
