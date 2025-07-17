import os
from glob import glob

label_dir = "./labels/val"  # update if needed

def remap_labels(path):
    for label_file in glob(os.path.join(path, "**", "*.txt"), recursive=True):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                if parts[0] == "15":
                    parts[0] = "0"  # raw_card
                elif parts[0] == "17":
                    parts[0] = "1"  # psa_slab
                else:
                    continue  # skip if not one of the two
                new_lines.append(" ".join(parts))

        if new_lines:
            with open(label_file, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            print(f"✅ Remapped: {label_file}")
        else:
            print(f"⚠️ Skipped: {label_file} (no valid labels)")

remap_labels(label_dir)