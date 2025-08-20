# live_translation.py
import cv2, time, joblib, numpy as np
import mediapipe as mp
from collections import deque, Counter

def run_live_translation():
    MODEL = "asl_svm.joblib"
    clf = joblib.load(MODEL)
    labels = list(clf.classes_)

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    face = mp_face.FaceDetection(min_detection_confidence=0.6)

    buf = deque(maxlen=5)
    subtitle = ""
    last_written = ""
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_ok = face.process(rgb).detections
        if not face_ok:
            cv2.putText(frame, "Bitte Gesicht zeigen...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Live-Übersetzung (ESC)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            pts -= pts[0]
            pts /= np.linalg.norm(pts[9]) + 1e-6
            feats = pts.flatten()
            probs = clf.predict_proba([feats])[0]
            idx = int(np.argmax(probs))
            char, prob = labels[idx], probs[idx]

            if prob >= 0.85 and char not in ["SPACE", "DEL", "NOTHING"]:
                now = time.time()
                if char != last_written or (now - last_time > 2):
                    subtitle += char + " "
                    subtitle = " ".join(subtitle.strip().split(" ")[-8:]) + " "
                    last_written = char
                    last_time = now

            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, subtitle.strip(), (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.imshow("Live-Übersetzung (ESC)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
