import itertools
import cv2
import numpy as np
import torch
import math
from collections import Counter

def sobel_from_luma(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    sobel = np.zeros((height, width), dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            val0 = int(gray[y - 1, x - 1])
            val1 = int(gray[y - 1, x])
            val2 = int(gray[y - 1, x + 1])
            val3 = int(gray[y, x - 1])
            val5 = int(gray[y, x + 1])
            val6 = int(gray[y + 1, x - 1])
            val7 = int(gray[y + 1, x])
            val8 = int(gray[y + 1, x + 1])

            gx = -val0 + -2 * val3 + -val6 + val2 + 2 * val5 + val8
            gy = -val0 + -2 * val1 + -val2 + val6 + 2 * val7 + val8

            mag = min(int((gx * gx + gy * gy) ** 0.5), 255)
            sobel[y, x] = mag
    return sobel

def calculate_border(sobel_img):
    _, binary = cv2.threshold(sobel_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(sobel_img, dtype=np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    border_mask = np.zeros_like(sobel_img, dtype=np.uint8)
    cv2.drawContours(border_mask, [largest_contour], -1, 255, thickness=2)
    return border_mask

def line_angle(line):
    x1, y1, x2, y2 = line
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180  # 0-180 degrees
    return angle

def filter_and_select_lines(lines, border_mask, vertical_thresh=15, horizontal_thresh=35, top_n=2):
    height, width = border_mask.shape
    vertical_lines = []
    horizontal_lines = []

    def line_angle(line):
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle) if abs(angle) <= 180 else 360 - abs(angle)

    def line_length(line):
        x1, y1, x2, y2 = line
        return np.hypot(x2 - x1, y2 - y1)

    def lines_far_enough(existing_lines, new_line, min_dist, orientation):
        x1n, y1n, x2n, y2n = new_line
        mid_new = np.array([(x1n + x2n) / 2, (y1n + y2n) / 2])
        for line in existing_lines:
            x1, y1, x2, y2 = line
            mid_existing = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            if orientation == 'horizontal':
                if abs(mid_new[1] - mid_existing[1]) < min_dist:
                    return False
            else:  # vertical
                if abs(mid_new[0] - mid_existing[0]) < min_dist:
                    return False
        return True

    def border_overlap_score(line):
        x1, y1, x2, y2 = line
        num_samples = 50
        count = 0
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            if 0 <= x < width and 0 <= y < height and border_mask[y, x] > 0:
                count += 1
        return count / num_samples  # score between 0 and 1

    # Collect and score vertical/horizontal lines
    scored_vertical = []
    scored_horizontal = []

    for line in lines:
        angle = line_angle(line)
        score = border_overlap_score(line)
        if score == 0:
            continue  # skip lines not overlapping with border
        if 90 - vertical_thresh <= angle <= 90 + vertical_thresh:
            scored_vertical.append((line, score))
        elif angle <= horizontal_thresh or angle >= 180 - horizontal_thresh:
            scored_horizontal.append((line, score))

    # Sort by score × length (stronger + longer lines come first)
    scored_vertical.sort(key=lambda x: x[1] * line_length(x[0]), reverse=True)
    scored_horizontal.sort(key=lambda x: x[1] * line_length(x[0]), reverse=True)

    def select_lines(scored_lines, min_dist, orientation):
        selected = []
        for line, _ in scored_lines:
            if len(selected) >= top_n:
                break
            if lines_far_enough(selected, line, min_dist, orientation):
                selected.append(line)
        return selected

    vertical_selected = select_lines(scored_vertical, min_dist=300, orientation='vertical')
    horizontal_selected = select_lines(scored_horizontal, min_dist=400, orientation='horizontal')

    print(f"Selected {len(vertical_selected)} vertical and {len(horizontal_selected)} horizontal lines")
    return vertical_selected + horizontal_selected


def calculate_lines(edge_img, border_mask=None):
    rho = 1
    theta = np.pi / 180
    threshold = 200
    min_line_length = 30
    max_line_gap = 50

    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return []

    lines = lines.reshape(-1, 4)
    # Filter and select only the most relevant lines (vertical + horizontal)
    filtered_lines = filter_and_select_lines(lines, border_mask)
    return filtered_lines
    #return lines

def extend_line(line, img_shape):
    x1, y1, x2, y2 = line
    height, width = img_shape[:2]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return (x1, y1, x2, y2)

    length = max(width, height) * 2
    mag = (dx**2 + dy**2)**0.5
    dx /= mag
    dy /= mag

    nx1 = int(x1 - dx * length)
    ny1 = int(y1 - dy * length)
    nx2 = int(x1 + dx * length)
    ny2 = int(y1 + dy * length)

    return (nx1, ny1, nx2, ny2)

def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)

def merge_close_points(points, radius=15):
    merged = []
    used = set()

    for i, p1 in enumerate(points):
        if i in used:
            continue
        cluster = [p1]
        used.add(i)

        for j, p2 in enumerate(points):
            if j in used:
                continue
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < radius:
                cluster.append(p2)
                used.add(j)

        # Average the cluster
        cluster = np.array(cluster)
        mean_point = tuple(np.mean(cluster, axis=0))
        merged.append(mean_point)

    return merged

def draw_intersections_on_image(image, intersections, color=(0, 255, 0), radius=4):
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1

    for idx, (x, y) in enumerate(intersections):
        center = (int(x), int(y))
        cv2.circle(img_vis, center, radius, color, -1)
        cv2.putText(img_vis, str(idx), (center[0] + 5, center[1] - 5),
                    font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return img_vis
def angle_between_vectors(v1, v2):
    # Normalize vectors
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # Compute dot product and clamp to avoid numerical issues
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def check_right_angles(pts, tolerance=15):
    # pts should be in order: [top-left, top-right, bottom-right, bottom-left]
    pts = np.array(pts)
    num_pts = len(pts)
    for i in range(num_pts):
        p0 = pts[i]
        p1 = pts[(i - 1) % num_pts]  # previous point
        p2 = pts[(i + 1) % num_pts]  # next point
        
        v1 = p1 - p0
        v2 = p2 - p0
        
        ang = angle_between_vectors(v1, v2)
        
        if not (90 - tolerance <= ang <= 90 + tolerance):
            return False
    return True

def calculate_corners(lines, width, height, min_aspect_ratio=0.65, max_aspect_ratio=0.8):
    intersections = []
    extended_lines = [extend_line(line, (height, width)) for line in lines]

    # Step 1: Find all pairwise intersections within bounds
    for i1 in range(len(extended_lines)):
        for i2 in range(i1 + 1, len(extended_lines)):
            pt = line_intersection(extended_lines[i1], extended_lines[i2])
            if pt is None:
                continue
            x, y = pt
            if 0 <= x < width and 0 <= y < height:
                print(f"Intersection found at: {pt}")
                intersections.append(pt)
    intersections = merge_close_points(intersections, radius=40)
    print(f"Total intersections found: {len(intersections)}")

    intersections_img = draw_intersections_on_image(cropped, intersections)
    cv2.namedWindow("All Intersections Labeled", cv2.WINDOW_NORMAL)
    cv2.imshow("All Intersections Labeled", intersections_img)
    cv2.waitKey(0)

    if len(intersections) < 4:
        return None

    min_area = float('inf')
    best_quad = None

    # Step 2: Try all combinations of 4 intersections
    for combo in itertools.combinations(intersections, 4):
        pts = np.array(combo, dtype=np.float32)
        pts = order_corners(pts)
        print(f"Checking quad: {pts}")

        if not cv2.isContourConvex(pts):
            print(f"Skipping non-convex quad: {pts}")
            continue
        if not check_right_angles(pts):
            print(f"Skipping quad with non-right angles: {pts}")
            continue
        #Compute bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(pts)
        if w == 0 or h == 0:
            print(f"Skipping degenerate quad with zero width/height: {pts}")
            continue
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            print(f"Skipping quad with aspect ratio {aspect_ratio} out of bounds: {pts}")
            continue

        area = cv2.contourArea(pts)
        if area <= 10000:  # Minimum area threshold
            print(f"Skipping quad with area too small: {area} for pts: {pts}")
            continue
        # print(f"Quad pts: {pts}")
        # print(f"Quad aspect ratio: {aspect_ratio}")
        # print(f"Quad area: {area}")
        if 0 < area < min_area:
            min_area = area
            best_quad = pts

    if best_quad is None:
        return None
    print(f"Found best quad with area: {min_area}")
    print(f"Best quad corners: {best_quad}")
    print(f"Best quad aspect ratio: {aspect_ratio}")
    return order_corners(best_quad)


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    # Return as a NumPy array (N, 2)
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def draw_lines_on_image(image, lines, color=(0, 0, 255), thickness=2):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for line in lines:
        x1, y1, x2, y2 = extend_line(line, image.shape)
        cv2.line(img_color, (x1, y1), (x2, y2), color, thickness)
    return img_color

def draw_corners_on_image(image, corners, color=(0, 255, 0)):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for (x, y) in corners:
        cv2.circle(img_color, (int(x), int(y)), 5, color, -1)
    return img_color

def get_top_colors(image, bucket=16, top_n=5):
    """
    image: numpy array in BGR format (OpenCV default)
    bucket: int, size of color bucket to group similar colors
    top_n: how many dominant colors to return
    
    Returns a list of bucketed colors (in RGB order) as tuples.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Bucket the colors by dividing each channel by bucket size
    bucketed = image_rgb // bucket
    
    # Reshape to list of pixels
    pixels = bucketed.reshape(-1, 3)
    
    # Convert pixels to tuples so they can be counted
    pixels_tuples = [tuple(p) for p in pixels]
    
    # Count frequency of each bucketed color
    counts = Counter(pixels_tuples)
    
    # Get the most common colors
    most_common = counts.most_common(top_n)
    
    # Convert back to full-scale color by multiplying bucket values by bucket size
    top_colors = [tuple(int(c * bucket + bucket // 2) for c in color) for color, _ in most_common]
    
    return top_colors

def extract_border_region(image, border_ratio=0.08):
    h, w = image.shape[:2]
    border_w = int(w * border_ratio)
    border_h = int(h * border_ratio)

    # Create mask for border (edges only)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border_h, :] = 255  # top
    mask[-border_h:, :] = 255  # bottom
    mask[:, :border_w] = 255  # left
    mask[:, -border_w:] = 255  # right

    return cv2.bitwise_and(image, image, mask=mask)

def create_color_mask(image, target_rgb, tolerance_h=10, tolerance_s=60, tolerance_v=60):
    """
    Create a mask of pixels within a tolerance of the target RGB color.
    This version works in HSV space for better perceptual accuracy.
    
    Args:
        image: Input BGR image (as from OpenCV).
        target_rgb: Target color in RGB format (e.g., from get_top_colors).
        tolerance_h/s/v: Hue, Saturation, and Value tolerances.

    Returns:
        Binary mask of pixels close to the target color.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    lower = np.array([
        max(int(target_hsv[0]) - tolerance_h, 0),
        max(int(target_hsv[1]) - tolerance_s, 0),
        max(int(target_hsv[2]) - tolerance_v, 0)
    ], dtype=np.uint8)

    upper = np.array([
        min(int(target_hsv[0]) + tolerance_h, 179),
        min(int(target_hsv[1]) + tolerance_s, 255),
        min(int(target_hsv[2]) + tolerance_v, 255)
    ], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Optional: smooth edges of the mask to reduce noise and harsh transitions
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


def enhance_masked_region(image, mask, boost_factor=1.7, saturation_boost=40):
    """
    Enhances regions in an image defined by a binary mask. 
    Increases brightness and saturation in masked areas.
    
    Args:
        image: Input BGR image.
        mask: Binary mask with white where enhancement is desired.
        boost_factor: Multiplier for brightness (Value channel).
        saturation_boost: Amount to increase Saturation.

    Returns:
        Enhanced BGR image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Brightness boost
    v[mask > 0] = np.clip(v[mask > 0] * boost_factor, 0, 255).astype(np.uint8)

    # Optional saturation boost
    s[mask > 0] = np.clip(s[mask > 0] + saturation_boost, 0, 255).astype(np.uint8)

    hsv_enhanced = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def morph_gradient(gray_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
    return grad

def visualize_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy()
    overlay[mask > 0] = (1 - alpha) * overlay[mask > 0] + alpha * np.array(color, dtype=np.uint8)
    return overlay.astype(np.uint8)

def fix_card_perspective(image_path, model_path, conf_thres=0.3, pad=0):
    global cropped
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
    model.conf = conf_thres

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None

    height, width = image.shape[:2]
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    if len(detections) == 0:
        print("No cards detected.")
        return None

    x1, y1, x2, y2, conf, cls = detections[0]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(width - 1, int(x2) + pad)
    y2 = min(height - 1, int(y2) + pad)
    cropped = image[y1:y2, x1:x2]
    cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Cropped Image", cropped)
    cv2.waitKey(0)

    border_only = extract_border_region(cropped)
    cv2.namedWindow("Border Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Border Image", border_only)
    cv2.waitKey(0)

    top_border_colors = get_top_colors(border_only, bucket=16, top_n=5)
    print("Top border colors (RGB):", top_border_colors)

    top_colors = get_top_colors(cropped, bucket=16, top_n=5)
    print("Top bucketed colors (approximate RGB):", top_colors)


    # hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # # Define yellow hue range in HSV
    # lower_yellow = np.array([15,  50,  50])   # More inclusive of duller/darker yellows
    # upper_yellow = np.array([45, 255, 255])   # Allow lighter/brighter yellows

    # # Create a binary mask where yellow is white and everything else is black
    # yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # # Optional: boost the yellow area to white and fade other regions
    # yellow_highlighted = cv2.bitwise_and(cropped, cropped, mask=yellow_mask)
    color1 = np.array(top_border_colors[1], dtype=np.uint8)
    color2 = np.array(top_border_colors[2], dtype=np.uint8)

    avg_border_color = top_colors[1]
    print("Average border color (RGB):", avg_border_color)
    border_mask = create_color_mask(cropped, avg_border_color)
    print("Mask non-zero pixels:", np.count_nonzero(border_mask))
    vis_mask = visualize_mask_overlay(cropped, border_mask)
    cv2.namedWindow("Mask Overlay", cv2.WINDOW_NORMAL)
    cv2.imshow("Mask Overlay", vis_mask)
    cv2.waitKey(0)

    sobel = np.clip(cv2.magnitude(
        cv2.Sobel(border_mask, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(border_mask, cv2.CV_64F, 0, 1, ksize=3)
    ), 0, 255).astype(np.uint8)
    cv2.namedWindow("Sobel Edge", cv2.WINDOW_NORMAL)
    cv2.imshow("Sobel Edge", sobel)
    cv2.waitKey(0)

    border = calculate_border(sobel)
    cv2.namedWindow("Border Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Border Mask", border)
    cv2.waitKey(0)

    lines = calculate_lines(border, border_mask)
    if len(lines) == 0:
        print("No lines detected.")
        return None

    print(f"Detected {len(lines)} lines")

    img_with_lines = draw_lines_on_image(sobel, lines)
    cv2.namedWindow("Lines Over Sobel", cv2.WINDOW_NORMAL)
    cv2.imshow("Lines Over Sobel", img_with_lines)
    cv2.waitKey(0)

    corners = calculate_corners(lines, cropped.shape[1], cropped.shape[0])
    if corners is None:
        print("Failed to detect 4 corners")
        return None

    corners = order_corners(corners)
    img_with_corners = draw_corners_on_image(cropped, corners)
    cv2.imshow("Corners on Image", img_with_corners)
    cv2.waitKey(0)

    # Match Pokémon card aspect ratio: 6.3 cm : 8.8 cm = 0.715
    out_h = 1024
    out_w = int(out_h * (6.3 / 8.8))  # ≈ 733

    src_pts = np.array(corners, dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    H_cv, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(cropped, H_cv, (out_w, out_h))

    return warped

# --- Run Example ---
warped_card = fix_card_perspective(
    image_path=r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\images\train2\ebay_jaivic_87\card_004_jaivic_87_raw.jpg",
    model_path=r"C:\Users\daforbes\Desktop\projects\tcg_scanner\yolov5\runs\train\card_detector_v22\weights\best.pt",
    conf_thres=0.3,
    pad=20
)

if warped_card is not None:
    cv2.imshow("Warped Card", warped_card)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("warped_card.png", warped_card)
else:
    print("Card perspective correction failed.")

import json
from PIL import Image
import imagehash

# Load previously computed hashes
with open(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\raw\card_hashes.json", "r") as f:
    hash_dict = json.load(f)

# Convert OpenCV BGR image to PIL RGB
warped_rgb = cv2.cvtColor(warped_card, cv2.COLOR_BGR2RGB)
warped_pil = Image.fromarray(warped_rgb)

# Generate perceptual hash
query_hash = imagehash.phash(warped_pil)

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
