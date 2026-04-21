import os
from pathlib import Path
from cluster_utils import list_images

DATASET_DIR = "../dataset"

def get_rotten_folders(root):
    return [f for f in os.listdir(root) if f.endswith("__Rotten")]

def main():
    rotten_folders = get_rotten_folders(DATASET_DIR)
    all_paths = []

    for folder in rotten_folders:
        folder_path = Path(DATASET_DIR) / folder
        imgs = list_images(folder_path)
        for p in imgs:
            all_paths.append((folder, p))

    print(f"Found {len(all_paths)} rotten images.")

    for old_folder, path in all_paths:
        fruit = old_folder.split("__")[0]
        grade = "C"  # All rotten → C
        new_folder = Path(DATASET_DIR) / f"{fruit}__{grade}"
        new_folder.mkdir(exist_ok=True)
        dest = new_folder / Path(path).name
        os.replace(path, dest)

    print("Rotten images assigned to grade C.")

if __name__ == "__main__":
    main()