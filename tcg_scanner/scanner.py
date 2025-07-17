import cv2
import torch
import numpy as np
from pathlib import Path

def detect_trading_card_yolo(image_path, model_path="best.pt", conf_threshold=0.25):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = conf_threshold

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Run inference
    results = model(image)
    detections = results.xyxy[0]  # Format: (x1, y1, x2, y2, conf, cls)

    if detections.shape[0] == 0:
        print("No cards detected by YOLOv5.")
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Use the highest-confidence detection (you can modify to pick others if needed)
    x1, y1, x2, y2, conf, cls = detections[0].cpu().numpy().astype(int)
    print(f"Detected bounding box: ({x1}, {y1}) to ({x2}, {y2}), Confidence: {conf:.2f}")

    # Draw bounding box for reference
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Prepare points for perspective correction (assuming rectangular card)
    src_pts = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ], dtype="float32")

    # Determine width and height
    width = x2 - x1
    height = y2 - y1
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(image, M, (width, height))

    # Show both
    cv2.imshow("Original with Bounding Box", image)
    cv2.imshow("Top-Down View", warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_trading_card_yolo(
        image_path="C:\\Users\\daforbes\\Desktop\\projects\\tcg_scanner\\cards\\zapdos.png",
        model_path="C:\\Users\\daforbes\\Desktop\\projects\\tcg_scanner\\yolov5\\runs\\train\\your_run_name\\weights\\best.pt"
    )
