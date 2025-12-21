import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model, load_model

# --------------------
# CONFIG
# --------------------
IMG_SIZE = 120
CLASS_THRESHOLD = 0.5
MIN_BOX_SIDE_PX = 15
MAX_BOX_AREA_RATIO = 0.6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

FULL_MODEL_PATH = os.path.join(MODEL_DIR, "facetracker_balanced.keras")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "facetracker_balanced.weights.h5")


# --------------------
# MODEL ARCHITECTURE
# --------------------
def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    vgg = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    for layer in vgg.layers:
        layer.trainable = False

    x = vgg.output

    # Classification head
    f1 = GlobalMaxPooling2D()(x)
    c1 = Dense(512, activation="relu")(f1)
    class_output = Dense(1, activation="sigmoid", name="class_output")(c1)

    # Bounding box regression head (YOLO-style)
    f2 = GlobalMaxPooling2D()(x)
    r1 = Dense(512, activation="relu")(f2)
    bbox_output = Dense(4, activation="sigmoid", name="bbox_output")(r1)

    return Model(inputs, [class_output, bbox_output])


# --------------------
# LOAD DETECTOR
# --------------------
def load_detector(model_path=None):
    """
    Loads the facetracker model safely.
    Priority:
    1. Full saved model (.keras or .h5)
    2. Architecture + weights (.weights.h5)
    
    Args:
        model_path: Optional path to model file. If provided, tries to load from this path first.
    """
    
    # If a specific path is provided, try to load it
    if model_path and os.path.exists(model_path):
        try:
            # Try loading as full model
            model = load_model(model_path, compile=False)
            print(f"✔ FaceTracker loaded from {model_path}")
            return model
        except Exception as e:
            print(f"⚠ Failed to load as full model from {model_path}: {e}")
            # Try loading as weights only
            try:
                model = build_model()
                model.load_weights(model_path)
                print(f"✔ FaceTracker loaded (weights) from {model_path}")
                return model
            except Exception as e2:
                print(f"⚠ Failed to load as weights from {model_path}: {e2}")
                # Continue to try default paths

    # 1. Full model (default paths)
    if os.path.exists(FULL_MODEL_PATH):
        try:
            model = load_model(FULL_MODEL_PATH, compile=False)
            print("✔ FaceTracker loaded (full model)")
            return model
        except Exception as e:
            print("⚠ Failed to load full model:", e)

    # 2. Weights only
    if os.path.exists(WEIGHTS_PATH):
        try:
            model = build_model()
            model.load_weights(WEIGHTS_PATH)
            print("✔ FaceTracker loaded (weights only)")
            return model
        except Exception as e:
            print("⚠ Failed to load weights:", e)

    print("❌ FaceTracker model not found")
    return None


# --------------------
# YOLO → XYXY
# --------------------
def yolo_to_xyxy(box, w, h):
    """
    Converts normalized YOLO-style bbox
    (xc, yc, bw, bh) → pixel (x1, y1, x2, y2)
    """
    xc, yc, bw, bh = box

    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    return x1, y1, x2, y2


# --------------------
# FACE DETECTION
# --------------------
def detect_face(model, frame):
    """
    Runs face detection on a single frame.
    Returns:
        bbox (x1,y1,x2,y2) or None
        confidence score
    """

    if model is None or frame is None or frame.size == 0:
        return None, 0.0

    H, W = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    inp = np.expand_dims(resized, axis=0)

    cls_pred, bbox_pred = model.predict(inp, verbose=0)
    score = float(cls_pred[0][0])

    # Confidence gate
    if score < CLASS_THRESHOLD:
        return None, score

    box = bbox_pred[0]
    x1, y1, x2, y2 = yolo_to_xyxy(box, W, H)

    # Sanity checks
    if (
        x2 <= x1 or
        y2 <= y1 or
        (x2 - x1) < MIN_BOX_SIDE_PX or
        (y2 - y1) < MIN_BOX_SIDE_PX or
        (x2 - x1) * (y2 - y1) > H * W * MAX_BOX_AREA_RATIO
    ):
        return None, score

    return (x1, y1, x2, y2), score
