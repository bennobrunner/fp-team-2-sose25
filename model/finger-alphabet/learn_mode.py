def run_learn_mode(window_name):
    import cv2, time, random, joblib, numpy as np
    import mediapipe as mp
    from collections import deque, Counter

    MODEL = "asl_svm.joblib"
    CAM_INDEX = 0
    LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    WIN = 5
    CONF_MIN = 0.7
    MATCH_THRESHOLD = 0.85

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    mp_face = mp.solutions.face_detection

    clf = joblib.load(MODEL)
    cap = cv2.VideoCapture(CAM_INDEX)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    def draw_center_text(f, t, color=(255, 255, 255)):
        h, w = f.shape[:2]
        size, _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        x = (w - size[0]) // 2
        cv2.putText(f, t, (x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    while True:
        target = random.choice(LETTERS)
        buf = deque(maxlen=WIN)
        matched = False
        start = time.time()

        while not matched:
            for _ in range(3): cap.grab()
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)

            face_ok = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).detections
            if not face_ok:
                draw_center_text(frame, "Gesicht zeigen...")
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) == 27: return
                continue

            cv2.putText(frame, f"Zeige: {target}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                       mp_style.get_default_hand_landmarks_style(),
                                       mp_style.get_default_hand_connections_style())
                pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                pts -= pts[0]; pts /= np.linalg.norm(pts[9]) + 1e-6
                feats = pts.flatten()
                proba = clf.predict_proba([feats])[0]
                idx = int(np.argmax(proba))
                pred, prob = clf.classes_[idx], float(proba[idx])
                if prob >= CONF_MIN:
                    buf.append(pred)

            if len(buf) == WIN:
                c = Counter(buf)
                best, cnt = c.most_common(1)[0]
                if best == target and cnt >= WIN // 2 + 1:
                    matched = True

            draw_center_text(frame, "Richtig!" if matched else "", color=(0, 255, 0) if matched else (255, 255, 255))
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == 27: return

        time.sleep(1)  # kurze Pause vor nächstem Ziel

    cap.release()
    hands.close()
    face.close()
