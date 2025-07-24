# from rembg import remove, new_session
# from PIL import Image
# import cv2
# import numpy as np
# import json
# import imagehash

# # --- Paths ---
# image_path = r"C:\Users\daforbes\Downloads\buyer-says-this-is-not-near-mint-what-grade-would-you-give-v0-msup3upd8n9f1.jpg"
# pad = 10  # Increased padding for better results

# # --- Step 1: Remove background using rembg ---
# print("🔄 Removing background with rembg...")

# # Load image
# input_image = Image.open(image_path)

# # Use isnet-general-use model for better object detection
# session = new_session('isnet-general-use')
# output_image = remove(input_image, session=session)

# # Convert to numpy array and separate channels
# img_array = np.array(output_image)
# rgb = img_array[:, :, :3]
# alpha = img_array[:, :, 3]

# # Convert RGB to BGR for OpenCV
# image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# print("✅ Background removed successfully")

# # --- Step 2: Create clean mask from alpha channel ---
# # Clean up alpha channel to create a binary mask
# mask = np.where(alpha > 128, 255, 0).astype(np.uint8)

# # Smooth the mask
# kernel = np.ones((5, 5), np.uint8)
# mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
# mask_smooth = cv2.GaussianBlur(mask_smooth, (5, 5), 0)

# print("✅ Mask created and smoothed")

# # --- Step 3: Find bounding box and crop ---
# # Find non-zero mask pixels
# ys, xs = np.where(mask_smooth > 0)
# if len(xs) == 0 or len(ys) == 0:
#     print("❌ Empty mask detected.")
#     exit()

# # Get bounding box coordinates
# x_min, x_max = xs.min(), xs.max()
# y_min, y_max = ys.min(), ys.max()

# # Apply padding with boundary checks
# x1 = max(x_min - pad, 0)
# x2 = min(x_max + pad, image_bgr.shape[1] - 1)
# y1 = max(y_min - pad, 0)
# y2 = min(y_max + pad, image_bgr.shape[0] - 1)

# # Crop both image and mask
# cropped_image = image_bgr[y1:y2, x1:x2]
# mask_cropped = mask_smooth[y1:y2, x1:x2]

# print(f"✅ Image cropped to {cropped_image.shape}")

# # --- Step 4: Display intermediate results (optional) ---
# cv2.namedWindow("Original with Background Removed", cv2.WINDOW_NORMAL)
# cv2.imshow("Original with Background Removed", image_bgr)

# cv2.namedWindow("Cropped Card", cv2.WINDOW_NORMAL)
# cv2.imshow("Cropped Card", cropped_image)

# cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
# cv2.imshow("Mask", mask_cropped)

# cv2.waitKey(1000)  # Show for 1 second

# # --- Step 5: Find corners of the card ---
# print("🔍 Finding card corners...")

# # Find contours of the card area
# contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# if not contours:
#     print("❌ No contours found in mask.")
#     exit()

# # Get the largest contour (should be the main card area)
# largest_contour = max(contours, key=cv2.contourArea)

# # Approximate the contour to get corner points
# epsilon = 0.01 * cv2.arcLength(largest_contour, True)
# approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# print(f"🔍 Found {len(approx)} corner points")

# def order_points(pts):
#     """Order points: top-left, top-right, bottom-right, bottom-left"""
#     pts = pts.astype(np.float32)
#     s = pts.sum(axis=1)
#     diff = np.diff(pts, axis=1).flatten()

#     ordered = np.zeros((4, 2), dtype="float32")
#     ordered[0] = pts[np.argmin(s)]      # top-left (smallest x+y)
#     ordered[2] = pts[np.argmax(s)]      # bottom-right (largest x+y)
#     ordered[1] = pts[np.argmin(diff)]   # top-right (smallest y-x)
#     ordered[3] = pts[np.argmax(diff)]   # bottom-left (largest y-x)
#     return ordered

# if len(approx) >= 4:
#     # If we have 4 or more points, use the 4 most extreme ones
#     if len(approx) == 4:
#         corner_points = approx.reshape(4, 2)
#     else:
#         # If more than 4 points, find the 4 extreme corners
#         points = approx.reshape(-1, 2)
        
#         # Find extreme points
#         top_left = points[np.argmin(points.sum(axis=1))]
#         bottom_right = points[np.argmax(points.sum(axis=1))]
#         top_right = points[np.argmin(points[:, 1] - points[:, 0])]
#         bottom_left = points[np.argmax(points[:, 1] - points[:, 0])]
        
#         corner_points = np.array([top_left, top_right, bottom_right, bottom_left])
    
#     src_pts = order_points(corner_points)
#     print("✅ Successfully found 4 corners of the card")
    
# else:
#     print(f"❌ Could not find enough corners. Found {len(approx)}, using bounding box")
    
#     # Fallback to bounding box
#     white_pixels = np.where(mask_cropped > 0)
#     if len(white_pixels[0]) == 0:
#         print("❌ No white pixels found in mask")
#         exit()
    
#     y_coords, x_coords = white_pixels
#     x_min, x_max = x_coords.min(), x_coords.max()
#     y_min, y_max = y_coords.min(), y_coords.max()
    
#     src_pts = np.array([
#         [x_min, y_min],    # top-left
#         [x_max, y_min],    # top-right
#         [x_max, y_max],    # bottom-right
#         [x_min, y_max]     # bottom-left
#     ], dtype="float32")

# # --- Step 6: Visualize corner points ---
# cropped_with_points = cropped_image.copy()
# colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # Green, Red, Blue, Yellow
# labels = ['TL', 'TR', 'BR', 'BL']

# for i, (point, color, label) in enumerate(zip(src_pts, colors, labels)):
#     x, y = int(point[0]), int(point[1])
#     cv2.circle(cropped_with_points, (x, y), 8, color, -1)
#     cv2.putText(cropped_with_points, label, (x + 10, y - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# cv2.namedWindow("Corner Points", cv2.WINDOW_NORMAL)
# cv2.imshow("Corner Points", cropped_with_points)
# cv2.waitKey(2000)

# # --- Step 7: Perspective transformation ---
# print("🔄 Applying perspective transformation...")

# # Define destination points (standard Pokémon card size ratio)
# out_h = 1024
# out_w = int(out_h * (6.3 / 8.8))  # ≈ 733

# dst_pts = np.array([
#     [0, 0],
#     [out_w - 1, 0],
#     [out_w - 1, out_h - 1],
#     [0, out_h - 1]
# ], dtype="float32")

# # Compute perspective transform
# M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# # Warp the cropped image
# warped_card = cv2.warpPerspective(cropped_image, M, (out_w, out_h))

# # Warp the mask to clean up the background
# warped_mask = cv2.warpPerspective(mask_cropped, M, (out_w, out_h))

# # Create final image with white background where mask is invalid
# warped_final = warped_card.copy()
# warped_final[warped_mask < 128] = [255, 255, 255]  # White background

# print("✅ Perspective transformation complete")

# # --- Step 8: Display and save results ---
# cv2.namedWindow("Final Warped Card", cv2.WINDOW_NORMAL)
# cv2.imshow("Final Warped Card", warped_final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Save the result
# cv2.imwrite("warped_card_rembg.png", warped_final)
# print("✅ Saved warped card as 'warped_card_rembg.png'")

# # --- Step 9: Hash matching (if hash dictionary exists) ---
# try:
#     with open(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\card_hashes.json", "r") as f:
#         hash_dict = json.load(f)
    
#     # Convert to PIL for hashing
#     warped_rgb = cv2.cvtColor(warped_final, cv2.COLOR_BGR2RGB)
#     warped_pil = Image.fromarray(warped_rgb)
    
#     # Generate perceptual hash
#     query_hash = imagehash.phash(warped_pil, hash_size=8)
    
#     # Find matches
#     distances = []
#     for key, stored_hash_str in hash_dict.items():
#         stored_hash = imagehash.hex_to_hash(stored_hash_str)
#         dist = query_hash - stored_hash
#         distances.append((key, dist))
    
#     # Sort by distance
#     distances.sort(key=lambda x: x[1])
    
#     # Show top 10 matches
#     print("\n🔍 Top 10 matches:")
#     for i, (key, dist) in enumerate(distances[:10]):
#         print(f"{i+1:>2}. {key} (distance={dist})")
        
# except FileNotFoundError:
#     print("⚠️  Hash dictionary not found, skipping card matching")
# except Exception as e:
#     print(f"⚠️  Error in hash matching: {e}")

# print("\n🎉 Processing complete!")