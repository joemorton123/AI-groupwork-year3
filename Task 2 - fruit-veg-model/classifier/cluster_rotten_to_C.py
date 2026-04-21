"""
Assign rotten fruit images to grade C folders.

Purpose
-------
This script scans dataset folders whose names end with "__Rotten", collects all image
files from those folders, and moves each image into a corresponding grade C folder
named using the pattern:

    <fruit>__C

Example:
    Apple__Rotten -> Apple__C

Assumptions
-----------
- The dataset root contains folders named like "<fruit>__Rotten".
- `cluster_utils.py` provides a `list_images(folder_path)` helper that returns image paths.
- All rotten fruit images are automatically assigned to grade C.

Side effects
------------
- Creates destination folders if they do not already exist.
- Moves files on disk using `os.replace`.
"""

import os
from pathlib import Path

from cluster_utils import list_images


# Root dataset directory containing fruit image folders.
DATASET_DIR = "../dataset"


def get_rotten_folders(root):
    """
    Return all folder names in the dataset root that represent rotten fruit images.

    Parameters
    ----------
    root : str or Path
        Path to the dataset root directory.

    Returns
    -------
    list[str]
        Folder names ending with "__Rotten".

    Notes
    -----
    Only folder names are returned, not full paths.
    """
    return [f for f in os.listdir(root) if f.endswith("__Rotten")]


def main():
    """
    Run the rotten-image reassignment process.

    Workflow
    --------
    1. Find all folders ending with "__Rotten".
    2. Collect all image paths from those folders.
    3. Derive the fruit name from the original folder name.
    4. Create a destination folder named "<fruit>__C".
    5. Move each rotten image into the corresponding grade C folder.

    Returns
    -------
    None

    Notes
    -----
    - All rotten images are treated as grade C without further analysis.
    - `os.replace` may overwrite an existing file if the destination path already exists.
    """
    rotten_folders = get_rotten_folders(DATASET_DIR)

    # Store pairs of (original_folder_name, image_path) for all rotten images.
    all_paths = []

    # Collect image paths from each rotten folder.
    for folder in rotten_folders:
        folder_path = Path(DATASET_DIR) / folder
        imgs = list_images(folder_path)

        for p in imgs:
            all_paths.append((folder, p))

    print(f"Found {len(all_paths)} rotten images.")

    # Move each rotten image into its corresponding grade C folder.
    for old_folder, path in all_paths:
        fruit = old_folder.split("__")[0]
        grade = "C"  # All rotten fruit is assigned to grade C.

        new_folder = Path(DATASET_DIR) / f"{fruit}__{grade}"
        new_folder.mkdir(exist_ok=True)

        dest = new_folder / Path(path).name
        os.replace(path, dest)

    print("Rotten images assigned to grade C.")


if __name__ == "__main__":
    main()