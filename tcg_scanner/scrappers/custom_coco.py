import os
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from labelme import utils

# Fixed categories
LABEL2ID = {
    "pokemon card": 0,
    "psa label": 1
}

CATEGORIES = [
    {"id": 0, "name": "pokemon card"},
    {"id": 1, "name": "psa label"}
]

def main(labelme_dir, output_file):
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
                print(f"Warning: skipping unknown label {label} in {json_path}")
                continue

            points = shape["points"]
            polygon = np.array(points).flatten().tolist()
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            min_x, min_y = min(x_coords), min(y_coords)
            max_x, max_y = max(x_coords), max(y_coords)
            width_box = max_x - min_x
            height_box = max_y - min_y

            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": LABEL2ID[label],
                "segmentation": [polygon],
                "bbox": [min_x, min_y, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0
            })

            ann_id += 1

        image_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    print(f"✅ COCO JSON saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing Labelme .json files")
    parser.add_argument("output_file", help="Path to output COCO .json file")
    args = parser.parse_args()

    main(r'', r'')
