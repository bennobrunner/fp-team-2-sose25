import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Eingabeverzeichnis
DATASET_DIR = "./data"

# Pfade für Outputs
OUT_CSV = "landmarks.csv"
OUT_LABELMAP = "label_map.json"

mp_hands = mp.solutions.hands

# Alle Bilddateien aus den Klassenordnern auflisten
def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        str(p) for d in sorted(Path(root).iterdir()) if d.is_dir()
        for p in d.iterdir() if p.suffix.lower() in exts
    ]

# Landmarks normalisieren (Spiegeln, Zentrieren, Skalieren)
def normalize_landmarks(lm, handed):
    if handed.lower().startswith("left"):
        lm[:, 0] = 1.0 - lm[:, 0]
    wrist = lm[0, :3]
    lm[:, :3] -= wrist
    min_xy = lm[:, :2].min(axis=0)
    max_xy = lm[:, :2].max(axis=0)
    diag = float(np.linalg.norm(max_xy - min_xy)) or 1.0
    lm[:, :3] /= diag
    return lm.astype(np.float32).reshape(-1)

# Einzelnes Bild verarbeiten und Landmarks extrahieren
def extract_from_image(img_path, hands_detector):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None, None
    hand_landmarks = res.multi_hand_landmarks[0]
    handedness_label = res.multi_handedness[0].classification[0].label
    lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    feat = normalize_landmarks(lm, handedness_label)
    return feat, handedness_label

# Mapping von Klassenname → ID erstellen
def build_label_map(root):
    classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
    label2id = {c: i for i, c in enumerate(classes)}
    return label2id

# Bilder laden, Landmarks extrahieren
def main():
    files = list_images(DATASET_DIR)
    label2id = build_label_map(DATASET_DIR)

    rows = []
    with mp_hands.Hands(
        static_image_mode=True,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        for fp in files:
            label = Path(fp).parent.name
            y = label2id[label]
            feat, handed = extract_from_image(fp, hands)
            if feat is None:
                continue
            row = {"path": fp, "label": label, "y": y, "handedness": handed}
            for i, v in enumerate(feat):
                row[f"f_{i:02d}"] = float(v)
            rows.append(row)

    # CSV + Labelmap speichern
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    with open(OUT_LABELMAP, "w", encoding="utf-8") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {i: c for c, i in label2id.items()}
        }, f, indent=2)

# Skript starten
if __name__ == "__main__":
    main()