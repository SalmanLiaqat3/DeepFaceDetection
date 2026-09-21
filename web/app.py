import os
import cv2
import pickle
import numpy as np
import base64
import subprocess
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from keras_facenet import FaceNet
from facetracker import load_detector, detect_face

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_DIR = os.path.join(DATA_DIR, "users")
EMB_PKL = os.path.join(DATA_DIR, "embeddings_facenet.pkl")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
ATT_DIR = os.path.join(DATA_DIR, "Attendance")

os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

IMG_SZ = (160, 160)
SIM_THRESHOLD = 0.60

# Colab consensus constants (exact match)
PER_FRAME_SIM = 0.60
CONSENSUS_FRAMES = 2
INSTANT_SIM_THRESH = 0.82
UNKNOWN_FRAMES_TO_BLOCK = 2

# ---------------- INIT ----------------
app = Flask(__name__)

FACETRACKER_WEIGHTS = os.path.join(BASE_DIR, "model", "facetracker_balanced.h5")
EARLY_CNN_WEIGHTS = os.path.join(BASE_DIR, "model", "early_cnn_detector.h5")


def safe_load_detector(path, label):
    """
    Helper to load a facetracker-style detector and log status.
    Returns the loaded model or None.
    """
    if not os.path.exists(path):
        print(f"WARNING: {label} weights not found: {path}")
        return None

    try:
        model = load_detector(path)
        if model is not None:
            print(f"[OK] {label} loaded successfully from {path}")
        else:
            print(f"WARNING: {label} failed to load from {path}")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load {label}: {e}")
        import traceback

        traceback.print_exc()
        return None


# Load detectors safely
FACE_DETECTOR = safe_load_detector(FACETRACKER_WEIGHTS, "Facetracker detector")
EARLY_CNN_DETECTOR = safe_load_detector(EARLY_CNN_WEIGHTS, "Early CNN detector")

if FACE_DETECTOR is None:
    print("Note: Face Detection will use Haar cascade (add model/facetracker_balanced.h5 for VGG-16).")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

embedder = FaceNet()

# load embeddings safely
MEANS = {}
if os.path.exists(EMB_PKL):
    db = pickle.load(open(EMB_PKL, "rb"))
    MEANS = db.get("mean", {})


# ---------------- HELPERS ----------------
def is_valid_image(img):
    return img is not None and isinstance(img, np.ndarray) and img.size > 0

def base64_to_image(base64_string):
    """Convert base64 data URL to OpenCV image"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

def image_to_base64(image):
    """Convert OpenCV image to base64 data URL"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


def detect_with_haar(image):
    """
    Simple Haarcascade-based face detector.
    Returns:
        bbox (x1, y1, x2, y2) or None
        pseudo-confidence (float or None)
    """
    if image is None or image.size == 0:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None, None

    # Pick the largest detected face
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    return (int(x), int(y), int(x + w), int(y + h)), 1.0  # pseudo confidence

def recognize_image(image):
    annotated = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        cv2.putText(annotated, "No face detected", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return "No Face", 0.0, "blocked_unknowns", annotated

    x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    crop = image[y:y+h, x:x+w]

    crop = cv2.resize(crop, IMG_SZ)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    emb = embedder.embeddings([rgb])[0]
    emb /= (np.linalg.norm(emb) + 1e-10)

    best_name = "Unknown"
    best_sim = 0.0

    for name, mean_emb in MEANS.items():
        sim = float(np.dot(emb, mean_emb))
        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_sim < SIM_THRESHOLD:
        best_name = "Unknown"
        reason = "blocked_unknowns"
        color = (0,0,255)
    else:
        reason = "accepted"
        color = (0,255,0)

    label = f"{best_name} {best_sim:.2f}"
    cv2.rectangle(annotated, (x,y),(x+w,y+h), color, 2)
    cv2.putText(annotated, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return best_name, best_sim, reason, annotated

    

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -------- ADD USER (NEW) --------
@app.route("/add_user", methods=["POST"])
def add_user():
    name = request.form.get("name")
    
    # Handle both file upload and base64 frame
    image = None
    if request.files.get("image"):
        file = request.files.get("image")
        img_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    elif request.form.get("frame"):
        # Handle base64 frame from camera
        frame_data = request.form.get("frame")
        image = base64_to_image(frame_data)

    if not name or not image or not is_valid_image(image):
        return jsonify({"error": "Name and valid image required"}), 400

    person_dir = os.path.join(USERS_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    # Count existing images to get next index
    existing = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    idx = len(existing) + 1
    img_path = os.path.join(person_dir, f"{name}_{idx:03d}.jpg")
    cv2.imwrite(img_path, image)

    return jsonify({
        "status": "saved",
        "path": img_path,
        "count": idx,
        "note": "Add more images, then rebuild embeddings"
    })

@app.route("/detect_face", methods=["POST"])
def detect_face_route():
    data = request.get_json()
    frame_data = data.get("frame") if data else None

    if not frame_data:
        return jsonify({"detected": False})

    image = base64_to_image(frame_data)

    if image is None or image.size == 0:
        return jsonify({"detected": False})

    # Use FaceTracker if loaded, otherwise fall back to Haar cascade
    if FACE_DETECTOR is not None:
        bbox, score = detect_face(FACE_DETECTOR, image)
    else:
        bbox, score = detect_with_haar(image)
        if score is None:
            score = 0.0

    annotated = image.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Face {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return jsonify({
        "detected": bbox is not None,
        "confidence": round(score, 3),
        "image": image_to_base64(annotated)
    })


@app.route("/detect_face_compare", methods=["POST"])
def detect_face_compare():
    """
    Run face detection with three different models on the same frame:
    - OpenCV Haarcascade
    - Early CNN detector
    - Facetracker detector
    """
    data = request.get_json()
    frame_data = data.get("frame") if data else None

    if not frame_data:
        return jsonify({"error": "No frame received"}), 400

    image = base64_to_image(frame_data)

    if image is None or image.size == 0:
        return jsonify({"error": "Invalid image data"}), 400

    # Haarcascade result
    haar_bbox, haar_conf = detect_with_haar(image)
    haar_annot = image.copy()
    if haar_bbox is not None:
        x1, y1, x2, y2 = haar_bbox
        cv2.rectangle(haar_annot, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            haar_annot,
            "Haar",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    # Early CNN result
    early_bbox, early_score = (None, 0.0)
    early_annot = image.copy()
    if EARLY_CNN_DETECTOR is not None:
        early_bbox, early_score = detect_face(EARLY_CNN_DETECTOR, image)
        if early_bbox is not None:
            x1, y1, x2, y2 = early_bbox
            cv2.rectangle(early_annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                early_annot,
                f"EarlyCNN {early_score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    # Facetracker result
    ft_bbox, ft_score = (None, 0.0)
    ft_annot = image.copy()
    if FACE_DETECTOR is not None:
        ft_bbox, ft_score = detect_face(FACE_DETECTOR, image)
        if ft_bbox is not None:
            x1, y1, x2, y2 = ft_bbox
            cv2.rectangle(ft_annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                ft_annot,
                f"Facetracker {ft_score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    return jsonify(
        {
            "haar": {
                "detected": haar_bbox is not None,
                "confidence": round(haar_conf, 3) if haar_conf is not None else None,
                "image": image_to_base64(haar_annot),
            },
            "early_cnn": {
                "loaded": EARLY_CNN_DETECTOR is not None,
                "detected": early_bbox is not None if EARLY_CNN_DETECTOR else False,
                "confidence": round(early_score, 3) if EARLY_CNN_DETECTOR else None,
                "image": image_to_base64(early_annot),
            },
            "facetracker": {
                "loaded": FACE_DETECTOR is not None,
                "detected": ft_bbox is not None if FACE_DETECTOR else False,
                "confidence": round(ft_score, 3) if FACE_DETECTOR else None,
                "image": image_to_base64(ft_annot),
            },
        }
    )

# -------- ADD USER FRAME (Auto-capture endpoint) --------
@app.route("/add_user_frame", methods=["POST"])
def add_user_frame():
    name = request.form.get("name")
    frame_data = request.form.get("frame")

    if not name or not frame_data:
        return jsonify({"saved": False})

    image = base64_to_image(frame_data)
    if image is None or image.size == 0:
        return jsonify({"saved": False})

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60)
    )

    if len(faces) == 0:
        return jsonify({"saved": False})

    # largest face (same as Colab)
    x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    face = image[y:y+h, x:x+w]

    if face.size == 0:
        return jsonify({"saved": False})

    face = cv2.resize(face, IMG_SZ)

    person_dir = os.path.join(USERS_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    idx = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) + 1
    cv2.imwrite(os.path.join(person_dir, f"{name}_{idx:03d}.jpg"), face)

    return jsonify({"saved": True, "count": idx})

# -------- REBUILD EMBEDDINGS --------
@app.route("/rebuild_embeddings", methods=["POST"])
def rebuild_embeddings():
    """
    Runs build_embeddings.py and reloads embeddings into memory
    """
    try:
        # Run the embedding builder using SAME python + venv
        subprocess.run(
            [sys.executable, "build_embeddings.py"],
            check=True,
            cwd=BASE_DIR
        )

        # Reload embeddings after rebuild
        global MEANS
        db = pickle.load(open(EMB_PKL, "rb"))
        MEANS = db.get("mean", {})

        return jsonify({
            "status": "success",
            "users": list(MEANS.keys())
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": "Embedding build failed",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to rebuild embeddings",
            "details": str(e)
        }), 500

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "No frames received"}), 400

    per_name_counts = {}
    per_name_sums = {}
    unknown_frame_count = 0
    max_sim_seen = 0.0
    max_sim_name = None
    last_annotated = None

    # Process ALL frames (not just the last one) - matches Colab logic
    for frame_data in data:
        frame = base64_to_image(frame_data)
        if not is_valid_image(frame):
            unknown_frame_count += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        annotated = frame.copy()

        if len(faces) == 0:
            unknown_frame_count += 1
            last_annotated = annotated
            continue

        x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            unknown_frame_count += 1
            continue

        crop = cv2.resize(crop, IMG_SZ)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        emb = embedder.embeddings([rgb])[0]
        emb /= (np.linalg.norm(emb) + 1e-10)

        best_name = None
        best_sim = 0.0
        for name, mean_emb in MEANS.items():
            sim = float(np.dot(emb, mean_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # Track frames meeting PER_FRAME_SIM threshold (CRITICAL: check best_name is not None)
        if best_sim >= PER_FRAME_SIM and best_name is not None:
            per_name_counts[best_name] = per_name_counts.get(best_name, 0) + 1
            per_name_sums[best_name] = per_name_sums.get(best_name, 0.0) + best_sim
        else:
            unknown_frame_count += 1

        if best_sim > max_sim_seen:
            max_sim_seen = best_sim
            max_sim_name = best_name

        # Display label: show "Unknown" if similarity below threshold (matches Colab semantics)
        display_name = best_name if (best_sim >= PER_FRAME_SIM and best_name is not None) else "Unknown"
        color = (0,255,0) if best_sim >= PER_FRAME_SIM and best_name is not None else (0,0,255)
        label = f"{display_name} {best_sim:.2f}"
        cv2.rectangle(annotated, (x,y),(x+w,y+h), color, 2)
        cv2.putText(annotated, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        last_annotated = annotated

        # Early exit: instant accept threshold
        if max_sim_seen >= INSTANT_SIM_THRESH:
            break

        # Early exit: too many unknown frames
        if unknown_frame_count >= UNKNOWN_FRAMES_TO_BLOCK:
            break

    # STRICT ACCEPTANCE LOGIC (matches Colab exactly)
    accepted = None
    
    # Rule 1: Consensus (2+ frames with same name above PER_FRAME_SIM)
    for name, count in per_name_counts.items():
        if count >= CONSENSUS_FRAMES:
            accepted = name
            break
    
    # Rule 2: Instant accept (single frame above INSTANT_SIM_THRESH)
    if not accepted and max_sim_seen >= INSTANT_SIM_THRESH and max_sim_name is not None:
        accepted = max_sim_name
    
    # If neither condition met → Unknown (strict rejection)

    # Save annotated image to results folder
    result_img_base64 = None
    result_img_url = None
    
    if last_annotated is not None and is_valid_image(last_annotated):
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # milliseconds
        filename = f"detection_{timestamp}.jpg"
        output_path = os.path.join(RESULT_DIR, filename)
        
        # Save the annotated image
        cv2.imwrite(output_path, last_annotated)
        
        # Convert to base64 for immediate display
        result_img_base64 = image_to_base64(last_annotated)
        
        # Generate URL for the saved image
        result_img_url = f"/static/results/{filename}"

    # Record attendance if accepted and not already in today's sheet
    already_in_attendance = False
    if accepted and accepted != "Unknown":
        csv_path = os.path.join(
            ATT_DIR, f"Attendance_{datetime.now().strftime('%d-%m-%Y')}.csv"
        )
        existing_names = set()
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                for line in f.readlines()[1:]:  # skip header
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if parts:
                        existing_names.add(parts[0].strip())
        if accepted in existing_names:
            already_in_attendance = True
        else:
            avg_sim = per_name_sums.get(accepted, max_sim_seen) / per_name_counts.get(accepted, 1)
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write("NAME,TIME,SIMILARITY\n")
                f.write(f"{accepted},{datetime.now().strftime('%H:%M:%S')},{avg_sim:.3f}\n")

    # Determine reason
    if accepted:
        reason = "consensus" if per_name_counts.get(accepted, 0) >= CONSENSUS_FRAMES else "instant_accept"
    else:
        reason = "blocked_unknowns"

    return jsonify({
        "result": accepted if accepted else "Unknown",
        "label": accepted if accepted else "Unknown",
        "similarity": round(max_sim_seen, 3),
        "reason": reason,
        "already_in_attendance": already_in_attendance,
        "image": result_img_base64,  # Base64 for immediate display
        "image_url": result_img_url   # URL to saved image in results folder
    })


@app.route("/api/attendance/today", methods=["GET"])
def attendance_today():
    """Return today's attendance entries only (from CSV)."""
    today_str = datetime.now().strftime("%d-%m-%Y")
    csv_path = os.path.join(ATT_DIR, f"Attendance_{today_str}.csv")
    entries = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            lines = f.readlines()
        # Skip header "NAME,TIME,SIMILARITY"
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                entries.append({
                    "name": parts[0].strip(),
                    "time": parts[1].strip(),
                    "similarity": parts[2].strip(),
                })
            elif len(parts) == 2:
                entries.append({
                    "name": parts[0].strip(),
                    "time": parts[1].strip(),
                    "similarity": "",
                })
    return jsonify({
        "date": today_str,
        "entries": entries,
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
