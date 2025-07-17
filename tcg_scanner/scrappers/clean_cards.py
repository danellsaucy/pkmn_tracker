import cv2
import numpy as np
from PIL import Image

def remove_white_background_and_crop(pil_image):
    # Convert PIL image to OpenCV format
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert threshold: make card dark, background white
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return pil_image

    # Get the largest contour (assume it's the card)
    largest = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest)

    # Crop to bounding box
    cropped = img[y:y+h, x:x+w]

    # Convert back to PIL image
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

def show_side_by_side(pil_before, pil_after):
    # Convert PIL images to OpenCV (BGR)
    before = cv2.cvtColor(np.array(pil_before), cv2.COLOR_RGB2BGR)
    after = cv2.cvtColor(np.array(pil_after), cv2.COLOR_RGB2BGR)

    # Resize to same height for comparison
    h = min(before.shape[0], after.shape[0])
    before = cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h))
    after = cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h))

    # Stack side by side
    combined = np.hstack((before, after))

    # Show window
    cv2.imshow("Original (left) vs Cleaned (right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

original = Image.open(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\downloaded_cards\151\sv3-5_en_003_std.jpg")
cleaned = remove_white_background_and_crop(original)
show_side_by_side(original, cleaned)
