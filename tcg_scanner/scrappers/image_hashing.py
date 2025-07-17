import os
import imagehash
from PIL import Image
import json

# Set this to your image folder root (with subfolders like /destined-rivals/)
ROOT_DIR = r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\downloaded_cards"
HASH_FILE_6x6 = "card_hashes10x10.json"
HASH_FILE_12x12 = "card_hashes16x16.json"

hash_dict_6x6 = {}
hash_dict_12x12 = {}

for set_name in os.listdir(ROOT_DIR):
    set_dir = os.path.join(ROOT_DIR, set_name)
    if not os.path.isdir(set_dir):
        continue

    for filename in os.listdir(set_dir):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(set_dir, filename)
            try:
                img = Image.open(image_path).convert("RGB")
                key = f"{set_name}/{filename}"
                
                # Compute both hash sizes
                hash6 = str(imagehash.phash(img, hash_size=10))
                hash12 = str(imagehash.phash(img, hash_size=16))
                
                hash_dict_6x6[key] = hash6
                hash_dict_12x12[key] = hash12

            except Exception as e:
                print(f"❌ Failed to hash {image_path}: {e}")

# Save the hashes to JSON files
with open(HASH_FILE_6x6, "w") as f6:
    json.dump(hash_dict_6x6, f6, indent=2)

with open(HASH_FILE_12x12, "w") as f12:
    json.dump(hash_dict_12x12, f12, indent=2)

print(f"✅ Hashed {len(hash_dict_6x6)} images → saved to {HASH_FILE_6x6} and {HASH_FILE_12x12}")
