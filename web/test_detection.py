import os
import cv2
import numpy as np
from facetracker import detect_face, load_detector, CLASS_THRESHOLD, IMG_SIZE

# Load model
weights = os.path.join('model', 'facetracker_balanced.weights.h5')
print("Loading model...")
model = load_detector(weights)
print("Model loaded!\n")

# Test folder - images are in test/images subfolder
test_folder = os.path.join("detection_aug_data", "test", "images")
if not os.path.exists(test_folder):
    print(f"Folder {test_folder} not found!")
    exit(1)

# Get first few images
jpg_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')]
if not jpg_files:
    print(f"No JPG files found in {test_folder}")
    exit(1)

print(f"Found {len(jpg_files)} images. Testing first 5...\n")

# Test with first 5 images
for i, img_name in enumerate(jpg_files[:5], 1):
    img_path = os.path.join(test_folder, img_name)
    print(f"Test {i}: {img_name}")
    
    # Load image
    test_img = cv2.imread(img_path)
    if test_img is None:
        print(f"  ❌ Failed to load image\n")
        continue
    
    print(f"  Image size: {test_img.shape}")
    
    # Get raw prediction for debugging
    h, w = test_img.shape[:2]
    rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    inp = np.expand_dims(resized, axis=0)
    cls, bbox = model.predict(inp, verbose=0)
    score = float(cls[0][0])
    x_n, y_n, w_n, h_n = bbox[0]
    print(f"  Detection score: {score:.3f} (threshold: {CLASS_THRESHOLD})")
    print(f"  Bbox normalized: x={x_n:.3f}, y={y_n:.3f}, w={w_n:.3f}, h={h_n:.3f}")
    
    # Detect face
    result = detect_face(test_img, model)
    
    if result:
        x1, y1, x2, y2 = result
        box_w = x2 - x1
        box_h = y2 - y1
        print(f"  [OK] Face detected!")
        print(f"  Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Box size: {box_w}x{box_h} pixels\n")
    else:
        print(f"  [NO] No face detected\n")

print("Testing complete!")
