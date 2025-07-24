from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import cv2
import torch
from PIL import Image  # Add this line

from rembg import new_session, remove

# --- Paths ---
config_file = r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\my_config.py'
checkpoint_file = r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\best_coco_segm_mAP_epoch_99.pth'
image_path = r"C:\Users\daforbes\Downloads\s-l1600 (1).jpg"
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

def clean_mask_with_rembg(cropped_image, mask_cropped):
    # Convert to RGB for rembg
    cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)
    
    # Get clean card mask from rembg
    session = new_session('isnet-general-use')
    rembg_result = remove(cropped_pil, session=session)
    rembg_mask = (np.array(rembg_result)[:, :, 3] > 128).astype(np.uint8) * 255
    
    # Use rembg mask to filter out outliers
    cleaned_mask = cv2.bitwise_and(mask_cropped, rembg_mask)
    
    return cleaned_mask

# --- Choose best scoring detection ---
best_idx = card_indices[scores[card_indices].argmax()]

mask = masks[best_idx].cpu().numpy().astype(np.uint8) * 255

# --- Smooth the mask ---
kernel = np.ones((5, 5), np.uint8)
mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_smooth = cv2.GaussianBlur(mask_smooth, (5, 5), 0)

# Find non-zero mask pixels
ys, xs = np.where(mask_smooth > 0)
if len(xs) == 0 or len(ys) == 0:
    print("❌ Empty mask detected.")
    exit()

# Get bounding box coordinates
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

# Apply padding with boundary checks
x1 = max(x_min - pad, 0)
x2 = min(x_max + pad, image.shape[1] - 1)
y1 = max(y_min - pad, 0)
y2 = min(y_max + pad, image.shape[0] - 1)

# Crop both image and mask
cropped_image = image[y1:y2, x1:x2]
mask_cropped = mask_smooth[y1:y2, x1:x2]
#mask_cropped = clean_mask_with_rembg(cropped_image, mask_cropped)
# --- Apply mask without changing background (keep original colors only where mask exists) ---
masked_cropped = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_cropped)

# Convert for display
cropped_bgr = cv2.cvtColor(masked_cropped, cv2.COLOR_RGB2BGR)

# --- Display intermediate results ---
# cv2.namedWindow("Original Cropped", cv2.WINDOW_NORMAL)
# cv2.imshow("Original Cropped", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

# cv2.namedWindow("Masked Cropped Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Masked Cropped Image", cropped_bgr)

# cv2.namedWindow("Smoothed Mask", cv2.WINDOW_NORMAL)
# cv2.imshow("Smoothed Mask", mask_cropped)

cv2.waitKey(1000)  # Show for 1 second, then continue

# --- FIND CORNERS OF WHITE AREAS IN MASK ---
# The mask already defines the card perfectly - white=card, black=background
# We need to find the corners of the white regions

# Find contours of the WHITE areas (card areas)
contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("❌ No contours found in mask.")
    exit()

# Get the largest contour (should be the main card area)
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour to get corner points
epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # More precise approximation
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

print(f"🔍 Found {len(approx)} corner points")

if len(approx) >= 4:
    # If we have 4 or more points, use the 4 most extreme ones
    if len(approx) == 4:
        corner_points = approx.reshape(4, 2)
    else:
        # If more than 4 points, find the 4 extreme corners
        points = approx.reshape(-1, 2)
        
        # Find extreme points
        top_left = points[np.argmin(points.sum(axis=1))]      # min(x+y)
        bottom_right = points[np.argmax(points.sum(axis=1))]  # max(x+y)  
        top_right = points[np.argmin(points[:, 1] - points[:, 0])]  # min(y-x)
        bottom_left = points[np.argmax(points[:, 1] - points[:, 0])]  # max(y-x)
        
        corner_points = np.array([top_left, top_right, bottom_right, bottom_left])
    
    # Order points properly: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]      # top-left (smallest x+y)
        ordered[2] = pts[np.argmax(s)]      # bottom-right (largest x+y)
        ordered[1] = pts[np.argmin(diff)]   # top-right (smallest y-x)
        ordered[3] = pts[np.argmax(diff)]   # bottom-left (largest y-x)
        return ordered
    
    src_pts = order_points(corner_points)
    print("✅ Successfully found 4 corners of the white card area")
    
else:
    print(f"❌ Could not find enough corners. Found {len(approx)}, need at least 4")
    print("Falling back to bounding box of white areas...")
    
    # Find white pixels
    white_pixels = np.where(mask_cropped > 0)
    if len(white_pixels[0]) == 0:
        print("❌ No white pixels found in mask")
        exit()
    
    y_coords, x_coords = white_pixels
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    src_pts = np.array([
        [x_min, y_min],    # top-left
        [x_max, y_min],    # top-right
        [x_max, y_max],    # bottom-right
        [x_min, y_max]     # bottom-left
    ], dtype="float32")

# --- Plot source points on cropped_bgr ---
cropped_bgr_with_points = cropped_bgr.copy()
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # Green, Red, Blue, Yellow
labels = ['TL', 'TR', 'BR', 'BL']

for i, (point, color, label) in enumerate(zip(src_pts, colors, labels)):
    x, y = int(point[0]), int(point[1])
    # Draw circle
    cv2.circle(cropped_bgr_with_points, (x, y), 8, color, -1)
    # Draw label
    cv2.putText(cropped_bgr_with_points, label, (x + 10, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Show image with points
# cv2.namedWindow("Source Points", cv2.WINDOW_NORMAL)
# cv2.imshow("Source Points", cropped_bgr_with_points)
# cv2.waitKey(2000)  # Show for 2 seconds

# --- Define destination points (standard Pokémon card size ratio) ---
out_h = 1024
out_w = int(out_h * (6.3 / 8.8))  # ≈ 733
dst_pts = np.array([
    [0, 0],
    [out_w - 1, 0],
    [out_w - 1, out_h - 1],
    [0, out_h - 1]
], dtype="float32")

# --- Compute perspective transform and warp ---
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Instead of warping the masked image (which has black areas), 
# warp the original cropped image first
warped_original = cv2.warpPerspective(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR), M, (out_w, out_h))

# Then warp the mask to know which areas are valid
mask_bgr = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2BGR)
warped_mask = cv2.warpPerspective(mask_bgr, M, (out_w, out_h))
warped_mask_single = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)

# Create a clean background (white) where mask is invalid
warped_final = warped_original.copy()
# Set background to white where mask is 0
warped_final[warped_mask_single < 128] = [255, 255, 255]  # White background

# Alternative: If you want transparent background (save as PNG)
# warped_rgba = cv2.cvtColor(warped_original, cv2.COLOR_BGR2BGRA)
# warped_rgba[warped_mask_single < 128] = [0, 0, 0, 0]  # Transparent
# warped_final = warped_rgba

# --- Show final result ---
cv2.imshow("Warped Card", warped_final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("warped_card.png", warped_final)

import json
from PIL import Image
import imagehash

# Load previously computed hashes
with open(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\card_hashes.json", "r") as f:
    hash_dict = json.load(f)

# Convert OpenCV BGR image to PIL RGB
warped_rgb = cv2.cvtColor(warped_final, cv2.COLOR_BGR2RGB)
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