from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import cv2
import torch

# --- Paths ---
config_file = r'C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\mask_point\my_config.py'
checkpoint_file = r'C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\mask_point\best_coco_segm_mAP_epoch_13.pth'
image_path = r"C:\Users\daforbes\Downloads\246845AD-23FF-4F9B-B2F0-682D50D09104.jpg"
pad = 0

# --- Initialize model ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# --- Load and infer ---
image = mmcv.imread(image_path, channel_order='rgb')
result = inference_detector(model, image)

# --- Extract predicted masks ---
masks = result.pred_instances.masks  # (N, H, W) -> BoolTensor
labels = result.pred_instances.labels  # (N,)
scores = result.pred_instances.scores  # (N,)

# --- Find 'pokemon card' detections (class 0) ---
card_indices = (labels == 0).nonzero(as_tuple=False).squeeze()

if card_indices.numel() == 0:
    print("❌ No 'pokemon card' detected.")
    exit()

# If only one index is found, ensure it's scalar
if card_indices.ndim == 0:
    card_indices = card_indices.unsqueeze(0)

# --- Choose best scoring detection ---
best_idx = card_indices[scores[card_indices].argmax()]
mask = masks[best_idx].cpu().numpy().astype(np.uint8) * 255


# --- Smooth the mask ---
kernel = np.ones((5, 5), np.uint8)
mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_smooth = cv2.GaussianBlur(mask_smooth, (5, 5), 0)

masked_image = cv2.bitwise_and(image, image, mask=mask_smooth)

# --- Create tight crop around the non-zero mask region ---
ys, xs = np.where(mask_smooth > 0)
if len(xs) == 0 or len(ys) == 0:
    print("❌ Empty mask detected.")
    exit()

x1, y1 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
x2, y2 = min(xs.max() + pad, image.shape[1] - 1), min(ys.max() + pad, image.shape[0] - 1)

# --- Crop masked region ---
cropped = masked_image[y1:y2, x1:x2]

cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
mask_cropped = mask_smooth[y1:y2, x1:x2]

# --- Display results ---
cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
cv2.imshow("Cropped Image", cropped_bgr)
cv2.namedWindow("Smoothed Mask", cv2.WINDOW_NORMAL)
cv2.imshow("Smoothed Mask", mask_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Step 1: Find the largest contour in the mask ---
contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("❌ No contours found.")
    exit()

largest_contour = max(contours, key=cv2.contourArea)

# --- Step 2: Approximate contour to polygon ---
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) != 4:
    print(f"❌ Found {len(approx)} corners instead of 4. Cannot apply perspective transform.")
    exit()

# --- Step 3: Order corners: top-left, top-right, bottom-right, bottom-left ---
def order_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left
    return ordered

src_pts = order_points(approx)

# --- Step 4: Define destination points (standard Pokémon card size ratio) ---
out_h = 1024
out_w = int(out_h * (6.3 / 8.8))  # ≈ 733
dst_pts = np.array([
    [0, 0],
    [out_w - 1, 0],
    [out_w - 1, out_h - 1],
    [0, out_h - 1]
], dtype="float32")

# --- Step 5: Compute perspective transform and warp ---
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(cropped_bgr, M, (out_w, out_h))

# --- Step 6: Show final result ---
cv2.imshow("Warped Card", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("warped_card.png", warped)

import json
from PIL import Image
import imagehash

# Load previously computed hashes
with open(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\card_hashes.json", "r") as f:
    hash_dict = json.load(f)

# Convert OpenCV BGR image to PIL RGB
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
warped_pil = Image.fromarray(warped_rgb)

# Generate perceptual hash
query_hash = imagehash.phash(warped_pil, hash_size=8)

# Collect all distances
distances = []

for key, stored_hash_str in hash_dict.items():
    stored_hash = imagehash.hex_to_hash(stored_hash_str)
    dist = query_hash - stored_hash  # Hamming distance
    distances.append((key, dist))

# Sort by distance (lowest = best match)
distances.sort(key=lambda x: x[1])

# Show top 10 matches
print("🔍 Top 10 matches:")
for i, (key, dist) in enumerate(distances[:10]):
    print(f"{i+1:>2}. {key} (distance={dist})")

