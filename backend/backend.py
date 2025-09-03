import time
import joblib
import numpy as np
from sanic import Sanic, json
from typing import Dict

app = Sanic("FP-Team2-Backend")

# -------------------- TUNABLES --------------------
MODEL_PATH = "data/asl_xgb.joblib"    # benutze wirklich das trainierte Modell
LABELMAP_PATH = "data/label_map.joblib"  # nur Fallback
CONF_THRESHOLD = 0.6                   # moderater als 0.8
PROB_EMA_ALPHA = 0.85                  # 0..1, höher = glatter (auf Probs)
STREAK_N = 3                           # gleiche Top-1 muss N-mal in Folge auftauchen
# --------------------------------------------------

# ---------- One-Euro-Filter für Landmark-Glättung ----------
class OneEuro:
    def __init__(self, freq=30.0, min_cutoff=1.2, beta=0.015, d_cutoff=1.0, dim=63):
        import math
        self.math = math
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dim = dim
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * self.math.pi * cutoff)
        te = 1.0 / max(self.freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def _sfilter(self, x, x_prev, a):
        return a * x + (1.0 - a) * x_prev

    def __call__(self, x, t=None):
        x = np.asarray(x, dtype=np.float32).reshape(self.dim)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = time.monotonic()
            return x

        t_now = time.monotonic() if t is None else t
        dt = max(t_now - self.t_prev, 1e-6)
        self.freq = 1.0 / dt
        self.t_prev = t_now

        # Geschwindigkeit schätzen und glätten
        dx = (x - self.x_prev) * self.freq
        ad = self._alpha(self.d_cutoff)
        dx_hat = self._sfilter(dx, self.dx_prev, ad)

        # adaptiver Cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)

        # eigentliche Glättung
        x_hat = self._sfilter(x, self.x_prev, a)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
# -----------------------------------------------------------

# Modell & Labels
model = joblib.load(MODEL_PATH)
if hasattr(model, "classes_"):
    LABELS = list(model.classes_)
else:
    # Fallback: Labelmap laden (Reihenfolge muss zu y-IDs passen!)
    lm = joblib.load(LABELMAP_PATH)
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    LABELS = [id2label[i] for i in range(len(id2label))]

# Pro-Sitzung Zustand (Glättung & Stabilisierung)
class SessionState:
    def __init__(self):
        self.oef = OneEuro(dim=63)           # Landmark-Glättung
        self.p_smooth = None                 # geglättete Probabilitäten
        self.prev_label = None
        self.streak = 0

SESSIONS: Dict[str, SessionState] = {}

def get_state(session_id: str) -> SessionState:
    st = SESSIONS.get(session_id)
    if st is None:
        st = SessionState()
        SESSIONS[session_id] = st
    return st

# ---------- Normalisierung exakt wie im Training ----------
def normalize_landmarks(lm: np.ndarray, handedness: str) -> np.ndarray:
    lm = np.asarray(lm, dtype=np.float32).copy()
    if handedness and handedness.lower().startswith("left"):
        lm[:, 0] = 1.0 - lm[:, 0]

    wrist = lm[0, :3]
    lm[:, :3] -= wrist

    min_xy = lm[:, :2].min(axis=0)
    max_xy = lm[:, :2].max(axis=0)
    diag = float(np.linalg.norm(max_xy - min_xy)) or 1.0
    lm[:, :3] /= diag

    return lm.reshape(1, -1).astype(np.float32)

def features_both_hands(lm: np.ndarray, handedness: str | None):
    """Falls Handedness unsicher: beide Varianten testen und die bessere nehmen."""
    if handedness:
        return [normalize_landmarks(lm, handedness)]
    return [normalize_landmarks(lm, "Right"),
            normalize_landmarks(lm, "Left")]

# ---------- Inferenz + Stabilisierung ----------
def infer_with_stability(state: SessionState, feats_list: list[np.ndarray]):
    best_idx = None
    best_prob = -1.0
    best_feat = None

    # Wähle beste Handedness-Variante
    for feats in feats_list:
        probs = model.predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        if probs[idx] > best_prob:
            best_prob = float(probs[idx])
            best_idx = idx
            best_feat = feats

    # Landmark-Glättung (auf Featurevektor)
    best_feat[0] = state.oef(best_feat[0])
    probs = model.predict_proba(best_feat)[0]

    # EMA auf Probabilitäten
    if state.p_smooth is None:
        state.p_smooth = probs
    else:
        state.p_smooth = PROB_EMA_ALPHA * state.p_smooth + (1.0 - PROB_EMA_ALPHA) * probs

    idx = int(np.argmax(state.p_smooth))
    conf = float(state.p_smooth[idx])

    # Streak-Debounce (reduziert Flackern zwischen ähnlichen Klassen)
    if LABELS[idx] == state.prev_label:
        state.streak += 1
    else:
        state.prev_label = LABELS[idx]
        state.streak = 1

    is_stable = (state.streak >= STREAK_N)
    char = LABELS[idx] if (conf >= CONF_THRESHOLD and is_stable) else ""

    return char, conf

# ----------------------- API -----------------------
@app.post("/fingers")
async def fingers(request):
    try:
        data = request.json or {}
        lm = np.array(data.get("landmarks") or [], dtype=np.float32)
        handedness = data.get("handedness")           # kann None sein
        session_id = str(data.get("session_id", "global"))  # optional vom Frontend setzen

        if lm.size == 0:
            return json({"character": "", "confidence": 0.0}, status=200)

        state = get_state(session_id)
        feats_list = features_both_hands(lm, handedness)
        char, conf = infer_with_stability(state, feats_list)

        return json({"character": char, "confidence": round(conf, 3)})

    except Exception as e:
        # defensiv: lieber leise ausfallen als crasht
        return json({"character": "", "error": str(e)}, status=200)

# Uncomment, if running locally
def __main__():
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    __main__()
