import os
from pathlib import Path


label_dir = Path(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\labels\ebay_psa")

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        cleaned = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 6:
                cleaned.append(' '.join(parts[:5]) + '\n')
            else:
                cleaned.append(line)

        with open(filepath, 'w') as f:
            f.writelines(cleaned)

print("Confidence scores removed.")
