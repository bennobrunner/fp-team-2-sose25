import joblib
import numpy as np
from sanic import Sanic, json

app = Sanic("TestApp")
model = joblib.load("../model/finger-alphabet/asl_svm.joblib")
labels = list(model.classes_)

@app.get("/moin")
async def moin(request):
    return json({"message": "Moin, moin!"})

@app.post("/fingers")
async def fingers(request):
    data = request.json
    landmarks = np.array(data.get("landmarks"), dtype=np.float32)
    handedness = data.get("handedness", "Left")  # Frontend soll 'Left'/'Right' mitsenden
    
    print(handedness)
 
    if len(landmarks) == 0:
        return json({"character": ""}, status=200)
 
    landmarks -= landmarks[0]
    landmarks /= np.linalg.norm(landmarks[9]) + 1e-6
 
  
    if handedness == "Right":
        landmarks[:, 0] *= -1.0
 
    landmarks = landmarks.reshape((1, -1))
 
    probabilities = model.predict_proba(landmarks)[0]
    idx = int(np.argmax(probabilities))
    char, prob = labels[idx], probabilities[idx]
 
    print(f"Predicted: {char}, Probability: {prob:.3f}")
 
    if prob < 0.7:
        char = ""
 
    return json({"character": char})

def __main__():
    app.run(host="localhost", port=8000)

if __name__ == "__main__":
    __main__()
