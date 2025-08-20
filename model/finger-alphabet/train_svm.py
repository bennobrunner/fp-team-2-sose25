# train_svm_cv.py
import json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

RANDOM_STATE = 42

df = pd.read_csv("landmarks.csv")
X = df.drop(columns=["label"]).values
y = df["label"].str.upper().values
classes = np.unique(y).tolist()

# Holdout-Test, der während des CV unangetastet bleibt
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=RANDOM_STATE))
])

param_grid = {
    "clf__C": [0.5, 1, 2, 5, 10],
    "clf__gamma": ["scale", 0.03, 0.1, 0.3, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    verbose=1
)
gs.fit(Xtr, ytr)

# Wahrscheinlichkeits-Kalibrierung (sigmoid/Platt). Bessere p-Werte als SVC(probability=True).
cal = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv=cv)
cal.fit(Xtr, ytr)

y_pred = cal.predict(Xte)
y_proba = cal.predict_proba(Xte)

print("\nBest params:", gs.best_params_)
print("\nClassification Report (Test):")
print(classification_report(yte, y_pred, digits=3))

cm = confusion_matrix(yte, y_pred, labels=classes, normalize="true")
top3 = top_k_accuracy_score(yte, y_proba, k=3, labels=classes)
print(f"Top-3 Accuracy (Test): {top3:.3f}")

# Artefakte speichern
joblib.dump(cal, "asl_svm_calibrated.joblib")
np.save("confusion_matrix_norm.npy", cm)
with open("model_meta.json", "w") as f:
    json.dump({
        "classes": classes,
        "best_params": gs.best_params_,
        "random_state": RANDOM_STATE
    }, f, indent=2)

print("Saved model to asl_svm_calibrated.joblib, confusion_matrix_norm.npy, model_meta.json")
