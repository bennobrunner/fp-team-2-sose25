import joblib
import numpy as np
from sanic import Sanic, json

app = Sanic("FP-Team2-Backend")

model = joblib.load("data/asl_svm.joblib")

# wenn du "label_map.joblib" hast:
labelmap = joblib.load("data/label_map.joblib")
id2label = {int(k): v for k, v in labelmap["id2label"].items()}
labels = [id2label[i] for i in range(len(id2label))]

# labels = list(model.classes_)  # fallback falls nur classes_ da ist

def normalize_landmarks(lm: np.ndarray, handedness: str) -> np.ndarray:
    if handedness.lower().startswith("left"):
        lm[:, 0] = 1.0 - lm[:, 0]

    wrist = lm[0, :3]
    lm[:, :3] = lm[:, :3] - wrist

    min_xy = lm[:, :2].min(axis=0)
    max_xy = lm[:, :2].max(axis=0)
    diag = float(np.linalg.norm(max_xy - min_xy)) or 1.0
    lm[:, :3] /= diag

    return lm.reshape(1, -1)

@app.post("/fingers")
async def fingers(request):
    data = request.json
    landmarks = np.array(data.get("landmarks"), dtype=np.float32)
    handedness = data.get("handedness", "Left")

    if len(landmarks) == 0:
        return json({"character": ""}, status=200)

    # richtige Normalisierung
    features = normalize_landmarks(landmarks, handedness)

    probabilities = model.predict_proba(features)[0]
    idx = int(np.argmax(probabilities))
    char, prob = labels[idx], probabilities[idx]

    print(f"Predicted: {char}, Probability: {prob:.3f}")

    if prob < 0.7:   # Konfidenz-Schwelle
        char = ""

    return json({"character": char})

# Uncomment, if running locally
def __main__():
    app.run(host="0.0.0.0", port=8080)
if __name__ == "__main__":
    __main__()
