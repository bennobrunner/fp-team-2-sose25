import time
import joblib
import numpy as np
from sanic import Sanic, json
from typing import Dict, Optional, List

app = Sanic("FP-Team2-Backend")

# -------------------- TUNABLES --------------------
MODEL_PATH = "data/asl_svm.joblib"      # Pfad zu deinem Modell
LABELMAP_PATH = "data/label_map.joblib" # nur Fallback, wenn classes_ fehlt
CONF_THRESHOLD = 0.6
PROB_EMA_ALPHA = 0.85
STREAK_N = 3
N_LM = 21
LM_DIM = 3
FEAT_DIM = N_LM * LM_DIM
# --------------------------------------------------

# ---------- One-Euro-Filter ----------
class OneEuro(object):
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

        dx = (x - self.x_prev) * self.freq
        ad = self._alpha(self.d_cutoff)
        dx_hat = self._sfilter(dx, self.dx_prev, ad)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = self._sfilter(x, self.x_prev, a)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
# ------------------------------------

# Modell & Labels
model = joblib.load(MODEL_PATH)
if hasattr(model, "classes_"):
    LABELS = list(model.classes_)
else:
    lm = joblib.load(LABELMAP_PATH)
    if isinstance(lm, dict) and "id2label" in lm:
        id2label = {int(k): v for k, v in lm["id2label"].items()}
        LABELS = [id2label[i] for i in range(len(id2label))]
    else:
        # letzter Ausweg: Keys sortieren
        id2label = {int(k): str(v) for k, v in lm.items()}
        LABELS = [id2label[i] for i in sorted(id2label.keys())]

# State pro Session
class SessionState(object):
    def __init__(self):
        self.oef = OneEuro(dim=FEAT_DIM)
        self.p_smooth = None
        self.prev_label = None
        self.streak = 0

SESSIONS = {}  # type: Dict[str, SessionState]

def get_state(session_id):
    st = SESSIONS.get(session_id)
    if st is None:
        st = SessionState()
        SESSIONS[session_id] = st
    return st

# --------- Utils ----------
def predict_proba_safe(model, X):
    """Gibt (n_samples, n_classes) zurück – auch wenn Modell kein predict_proba hat."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)  # (n, C) oder (n,) bei binär
        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            # binärer Fall -> in zwei Spalten konvertieren
            d = np.vstack([-d, d]).T
        # Softmax
        d -= np.max(d, axis=1, keepdims=True)
        e = np.exp(d)
        p = e / np.sum(e, axis=1, keepdims=True)
        return p.astype(np.float32)
    # Fallback: one-hot aus predict
    y = model.predict(X)
    classes = list(getattr(model, "classes_", LABELS))
    k = len(classes)
    P = np.full((len(y), k), 1.0 / k, dtype=np.float32)
    for i, yi in enumerate(y):
        try:
            idx = classes.index(yi)
        except ValueError:
            idx = int(yi) if isinstance(yi, (int, np.integer)) else 0
        P[i, :] = 0.0
        P[i, idx] = 1.0
    return P

def normalize_landmarks(lm, handedness):
    lm = np.asarray(lm, dtype=np.float32)
    if lm.size != FEAT_DIM:
        lm = lm.reshape(-1, LM_DIM)
    if lm.shape != (N_LM, LM_DIM):
        # unförmige Eingabe -> lieber sauber abbrechen
        raise ValueError("Landmarks müssen (21,3) sein, bekommen: %r" % (lm.shape,))
    lm = lm.copy()

    if handedness and isinstance(handedness, str) and handedness.lower().startswith("left"):
        lm[:, 0] = 1.0 - lm[:, 0]

    wrist = lm[0, :3]
    lm[:, :3] -= wrist

    min_xy = lm[:, :2].min(axis=0)
    max_xy = lm[:, :2].max(axis=0)
    diag = float(np.linalg.norm(max_xy - min_xy)) or 1.0
    lm[:, :3] /= diag

    return lm.reshape(1, -1).astype(np.float32)

def features_both_hands(lm, handedness):
    if handedness:
        return [normalize_landmarks(lm, handedness)]
    return [normalize_landmarks(lm, "Right"),
            normalize_landmarks(lm, "Left")]

def infer_with_stability(state, feats_list):
    # beste Handedness-Variante wählen
    best_prob = -1.0
    best_feat = None
    for feats in feats_list:
        probs = predict_proba_safe(model, f
