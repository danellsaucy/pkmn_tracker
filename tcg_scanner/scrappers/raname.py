import os

def append_folder_name_to_files(path):
    # Get the folder name from the path
    folder_name = os.path.basename(os.path.normpath(path))

    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)

        if not os.path.isfile(full_path):
            continue  # Skip subdirectories

        name, ext = os.path.splitext(filename)

        if folder_name in name:
            continue  # Skip if folder name is already in file name

        new_name = f"{name}_{folder_name}{ext}"
        new_path = os.path.join(path, new_name)

        os.rename(full_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

# Example usage:
# append_folder_name_to_files("/path/to/your/folder")
if __name__ == "__main__":
    # Replace with your actual path
    append_folder_name_to_files(r"C:\Users\daforbes\Desktop\projects\tcg_scanner\dataset\labels\ebay_sarveyresells")