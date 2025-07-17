import torch
import cv2
import numpy as np
import imagehash
from PIL import Image
import json

HASH_DB_PATH = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\raw\card_hashes.json"  # Your precomputed phash JSON database

def detect_and_match_card(image_path, model_path, conf_threshold=0.3, pad=10):
    # Load hash database
    with open(HASH_DB_PATH, "r") as f:
        hash_dict = json.load(f)

    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.conf = conf_threshold

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image.")
        return

    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    if len(detections) == 0:
        print("❌ No card detected.")
        return

    # Use the first detection
    x1, y1, x2, y2, conf, cls = detections[0]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(img.shape[1], int(x2) + pad)
    y2 = min(img.shape[0], int(y2) + pad)

    # Crop card and convert to PIL
    card_crop = img[y1:y2, x1:x2]
    pil_card = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))

    # Compute perceptual hash
    query_hash = imagehash.phash(pil_card)

    # Compare with database
    distances = []
    for key, stored_hash_str in hash_dict.items():
        stored_hash = imagehash.hex_to_hash(stored_hash_str)
        dist = query_hash - stored_hash
        distances.append((key, dist))

    # Sort and get top 5
    top_matches = sorted(distances, key=lambda x: x[1])[:5]

    print(f"📌 Top 5 matches for detected card (phash = {str(query_hash)}):")
    for i, (name, dist) in enumerate(top_matches, 1):
        print(f"{i}. {name} (distance={dist})")

    # Optional: show detected image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Detected Card", img)
    cv2.imshow("Cropped Card", card_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
detect_and_match_card(
    image_path=r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\images\train\card_081_raw_ebay_rozo17.jpg",
    model_path=r"C:\Users\daforbes\Desktop\projects\tcg_scanner\yolov5\runs\train\card_detector_v22\weights\best.pt"
)
