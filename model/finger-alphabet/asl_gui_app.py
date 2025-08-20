import tkinter as tk
import threading, time, joblib, cv2, random, numpy as np
from PIL import Image, ImageTk
import mediapipe as mp

# === Root starten ===
root = tk.Tk()
root.title("ASL Translator App")
root.geometry("960x600")
root.resizable(False, False)

# === GUI Elemente ===
video_label = tk.Label(root)
video_label.pack()

status_label = tk.Label(root, text="Modell wird geladen...", font=("Arial", 16))
status_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack()

# === Statusvariablen ===
mode = tk.StringVar(value="loading")
subtitle = ""
current_letter = ""
result_text = ""
last_written = ""
last_time = 0
last_result_time = 0
filter_start_time = None
filter_duration = 2.0
filter_fade_time = 0.5
next_letter_delay = 1.0
last_correct_time = 0
show_confidence_info = False

current_candidate = None
candidate_start_time = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

clf = None
labels = []
IGNORE_LABELS = {"DEL", "SPACE", "NOTHING"}
hands = None
face = None

# === UI Funktionen ===
def show_buttons():
    tk.Button(button_frame, text="Live-Übersetzung starten", font=("Arial", 12), command=start_translation).pack(side="left", padx=10)
    tk.Button(button_frame, text="Lernmodus starten", font=("Arial", 12), command=start_learning).pack(side="left", padx=10)
    tk.Button(button_frame, text="Beenden", font=("Arial", 12), command=on_quit).pack(side="left", padx=10)

def hide_buttons():
    for widget in button_frame.winfo_children():
        widget.destroy()

def reset_menu():
    global subtitle, last_written, last_time, result_text
    mode.set("menu")
    subtitle = ""
    last_written = ""
    last_time = 0
    result_text = ""
    status_label.config(text="Modus wählen:")
    show_buttons()
    video_label.config(image="")  # Kamera ausblenden

def start_translation():
    global subtitle, last_written, last_time
    mode.set("translate")
    subtitle = ""
    last_written = ""
    last_time = 0
    status_label.config(text="Live-Übersetzung aktiv")
    hide_buttons()

def start_learning():
    global current_letter, result_text, last_correct_time
    mode.set("learn")
    current_letter = random.choice(labels)
    result_text = ""
    last_correct_time = time.time() - next_letter_delay
    status_label.config(text=f"Zeige: {current_letter}")
    hide_buttons()

def apply_green_filter(frame, alpha):
    overlay = frame.copy()
    green = np.full_like(overlay, (0, 255, 0))
    return cv2.addWeighted(overlay, 1 - alpha, green, alpha, 0)

def draw_text_with_background(frame, text, pos, font, scale, text_color, bg_color, thickness, pad=10):
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - pad, y - h - pad), (x + w + pad, y + pad), bg_color, -1)
    cv2.putText(frame, text, pos, font, scale, text_color, thickness)

# === Kamerabild aktualisieren ===
def update_frame():
    global subtitle, last_written, last_time, current_letter, result_text, last_result_time
    global filter_start_time, last_correct_time, current_candidate, candidate_start_time

    if mode.get() not in {"translate", "learn"}:
        root.after(10, update_frame)
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = time.time()

    if clf and hands and face:
        face_ok = face.process(rgb).detections
        res = hands.process(rgb)

        if mode.get() == "translate":
            if not face_ok:
                draw_text_with_background(frame, "Bitte Gesicht zeigen...", (30, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), (0, 0, 0), 2)
            elif res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                pts -= pts[0]
                pts /= np.linalg.norm(pts[9]) + 1e-6
                feats = pts.flatten()
                probs = clf.predict_proba([feats])[0]
                idx = int(np.argmax(probs))
                char, prob = clf.classes_[idx], probs[idx]

                if char.upper() not in IGNORE_LABELS and prob >= 0.85:
                    if current_candidate != char:
                        current_candidate = char
                        candidate_start_time = now
                    elif now - candidate_start_time >= 1.0 and now - last_time >= 1.0:
                        subtitle += char + " "
                        subtitle = " ".join(subtitle.strip().split(" ")[-8:]) + " "
                        last_written = char
                        last_time = now
                        current_candidate = None
                        candidate_start_time = 0
                else:
                    current_candidate = None
                    candidate_start_time = 0

                if show_confidence_info:
                    draw_text_with_background(
                        frame,
                        f"{char.upper()} ({prob*100:.1f}%)",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        (0, 0, 0),
                        2
                    )

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
                )

            draw_text_with_background(frame, subtitle.strip(), (30, 450),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), (0, 0, 0), 2)

        elif mode.get() == "learn":
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                pts -= pts[0]
                pts /= np.linalg.norm(pts[9]) + 1e-6
                feats = pts.flatten()
                probs = clf.predict_proba([feats])[0]
                idx = int(np.argmax(probs))
                pred, prob = clf.classes_[idx], probs[idx]

                if pred.upper() not in IGNORE_LABELS and prob >= 0.85:
                    if pred == current_letter:
                        if now - last_correct_time > next_letter_delay:
                            result_text = "Richtig!"
                            current_letter = random.choice(labels)
                            status_label.config(text=f"Zeige: {current_letter}")
                            last_result_time = now
                            filter_start_time = now
                            last_correct_time = now
                    else:
                        if now - last_correct_time > next_letter_delay:
                            result_text = f"Falsch ({pred})"

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
                )

            draw_text_with_background(frame, result_text, (30, 450),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 0) if result_text == "Richtig!" else (0, 0, 255),
                                      (0, 0, 0), 2)

            if filter_start_time:
                dt = now - filter_start_time
                if dt <= filter_duration:
                    alpha = min(1.0, max(0.0,
                                         dt / filter_fade_time if dt < filter_fade_time
                                         else (filter_duration - dt) / filter_fade_time))
                    frame = apply_green_filter(frame, alpha)
                else:
                    filter_start_time = None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
    video_label.configure(image=img)
    video_label.image = img
    root.after(10, update_frame)

# === Ladeanimation ===
def animate_loading():
    if mode.get() == "loading":
        current = status_label.cget("text")
        if current.endswith("..."):
            status_label.config(text="Modell wird geladen")
        else:
            status_label.config(text=current + ".")
        root.after(500, animate_loading)

# === Modell laden ===
def load_model():
    global clf, labels, hands, face
    clf = joblib.load("asl_svm.joblib")
    labels = sorted([l for l in clf.classes_ if l.isalpha() and l.upper() not in IGNORE_LABELS])
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
    face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
    mode.set("menu")
    status_label.config(text="Modus wählen:")
    show_buttons()

# === Schließen ===
def on_quit():
    global cap
    if cap and cap.isOpened():
        cap.release()
    root.destroy()

# === Tasteneingaben ===
def on_key(event):
    global show_confidence_info
    if event.keysym == "Escape" and mode.get() != "menu":
        reset_menu()
    elif event.keysym.lower() == "i":
        show_confidence_info = not show_confidence_info

# === Start Setup ===
root.bind("<Key>", on_key)
root.protocol("WM_DELETE_WINDOW", on_quit)
animate_loading()
threading.Thread(target=load_model, daemon=True).start()
update_frame()
root.mainloop()
