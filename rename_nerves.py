import os
import re

# ==============================
# SET ROOT DIRECTORY HERE
# ==============================
ROOT_DIR = r"T:\Orthopaedics\Lab Imaging Data\mmazur\DeepAxon\DA Data\N20 Analysis\Control"

# Set to False after confirming output
DRY_RUN = False


def zero_pad_id(folder_name):
    """
    Converts:
    21G-20-2 -> 21G_20_02
    21G-20-02 -> 21G_20_02
    """
    parts = re.split(r'[-_]', folder_name)
    if len(parts) == 3:
        parts[2] = parts[2].zfill(2)
        return "_".join(parts)
    return folder_name.replace("-", "_")


def rename_files_in_folder(folder_path, new_folder_name):
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".tif"):

                old_path = os.path.join(subdir, file)

                match = re.search(r'40X(\d+)(_seg)?', file, re.IGNORECASE)
                if not match:
                    continue

                img_number = match.group(1)
                seg_part = match.group(2) if match.group(2) else ""

                new_filename = f"{new_folder_name}_40x_{img_number}{seg_part}.tif"
                new_path = os.path.join(subdir, new_filename)

                print(f"FILE: {file}  →  {new_filename}")

                if not DRY_RUN:
                    os.rename(old_path, new_path)


# ==============================
# MAIN LOOP
# ==============================

for item in os.listdir(ROOT_DIR):
    item_path = os.path.join(ROOT_DIR, item)

    if os.path.isdir(item_path):
        new_folder_name = zero_pad_id(item)
        new_folder_path = os.path.join(ROOT_DIR, new_folder_name)

        if item != new_folder_name:
            print(f"FOLDER: {item}  →  {new_folder_name}")
            if not DRY_RUN:
                os.rename(item_path, new_folder_path)
            item_path = new_folder_path

        rename_files_in_folder(item_path, new_folder_name)

print("\nDone.")
if DRY_RUN:
    print("DRY RUN MODE — No files were actually renamed.")
