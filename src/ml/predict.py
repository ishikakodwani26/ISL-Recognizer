
from pathlib import Path
import pickle

import numpy as np
from tensorflow.keras.models import load_model

from .preprocessing import preprocess_keypoints


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "isl_bilstm_model.keras"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

model = load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def predict_sign(keypoints):
    """Predict an ISL word from raw keypoints shaped (frames, 75, 3)."""

    processed = preprocess_keypoints(keypoints)
    batch = np.expand_dims(processed, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]

    class_id = int(np.argmax(probabilities))
    word = str(label_encoder.inverse_transform([class_id])[0])
    confidence = float(probabilities[class_id])

    return {
        "word": word,
        "confidence": confidence,
        "class_id": class_id,
    }