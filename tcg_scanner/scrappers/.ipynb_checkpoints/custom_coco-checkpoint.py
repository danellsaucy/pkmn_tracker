import os
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from labelme import utils

# Define class mappings
LABEL2ID = {
    "pokemon card": 0,
    "psa label": 1
}

CATEGORIES = [
    {"id": 0, "name": "pokemon card"},
    {"id": 1, "name": "psa label"}
]

def expand_bbox(x, y, w, h, img_width, img_height, padding=15):
    """Expand the bbox with padding and clip to image size."""
    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    w_new = min(img_width - x_new, w + 2 * padding)
    h_new = min(img_height - y_new, h + 2 * padding)
    return [x_new, y_new, w_new, h_new]

def main(labelme_dir, output_file, padding=15):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    image_id = 0
    ann_id = 0

    label_files = glob.glob(os.path.join(labelme_dir, "*.json"))

    for json_path in tqdm(label_files, desc="Converting"):
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        img_path = os.path.join(labelme_dir, label_data["imagePath"])
        img = Image.open(img_path)
        width, height = img.size

        coco_output["images"].append({
            "file_name": label_data["imagePath"],
            "height": height,
            "width": width,
            "id": image_id
        })

        for shape in label_data["shapes"]:
            label = shape["label"]
            if label not in LABEL2ID:
                print(f"⚠️ Skipping unknown label '{label}' in {json_path}")
                continue

            points = shape["points"]
            polygon = np.array(points).flatten().tolist()
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            min_x, min_y = min(x_coords), min(y_coords)
            max_x, max_y = max(x_coords), max(y_coords)
            box_w = max_x - min_x
            box_h = max_y - min_y

            padded_bbox = expand_bbox(min_x, min_y, box_w, box_h, width, height, padding)

            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": LABEL2ID[label],
                "segmentation": [polygon],
                "bbox": padded_bbox,
                "area": padded_bbox[2] * padded_bbox[3],
                "iscrowd": 0
            })

            ann_id += 1

        image_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"\n✅ COCO-format JSON saved to: {output_file}")

if __name__ == "__main__":
    input_dir = input("📁 Enter path to LabelMe folder: ").strip()
    output_file = input("💾 Enter path for output COCO JSON file: ").strip()
    padding_str = input("🧱 Enter padding in pixels (default 15): ").strip()

    try:
        padding = int(padding_str) if padding_str else 15
    except ValueError:
        print("❌ Invalid padding value. Using default 15.")
        padding = 15

    main(input_dir, output_file, padding)
