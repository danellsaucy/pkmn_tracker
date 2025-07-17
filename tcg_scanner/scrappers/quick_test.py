from labelme2coco import convert
import os

input_dir = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\images\val"
output_path = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\val.json"

convert(input_dir, output_path)
print(f"✅ COCO JSON saved to {output_path}")