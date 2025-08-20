from sanic import Sanic, json
import cv2, time, joblib, numpy as np

app = Sanic("TestApp")
model = joblib.load("../model/finger-alphabet/asl_svm.joblib")
labels = list(model.classes_)

@app.get("/moin")
async def moin(request):
    return json({"message": "Moin, moin!"})

@app.post("/fingers")
async def fingers(request):
    data = request.json
    landmarks = data.get("landmarks")
    landmarks = np.array(landmarks, dtype=np.float32)

    landmarks = landmarks.reshape((1, -1))

    probabilities = model.predict_proba(landmarks)[0]

    idx = int(np.argmax(probabilities))
    char, prob = labels[idx], probabilities[idx]

    print(f"Predicted: {char}, Probability: {prob:.3f}")
    return json({"character": char})

def __main__():
    app.run(host="localhost", port=8000)

if __name__ == "__main__":
    __main__()
