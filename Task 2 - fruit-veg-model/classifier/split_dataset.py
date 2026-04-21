"""
Create train, validation, and test splits from a graded image dataset.

Purpose
-------
This script copies images from a labelled dataset into a standard machine-learning
split structure:

    ../dataset_split/train/<grade>
    ../dataset_split/val/<grade>
    ../dataset_split/test/<grade>

It supports two source folder layouts:

1. Fruit-and-grade folders
   Example:
       Apple__A
       Banana__B
       Orange__C

2. Grade-only folders
   Example:
       A
       B
       C

This allows the same splitting script to be reused for both the original dataset
and the refined dataset produced after manual or model-assisted relabelling.

Usage
-----
To switch between the first-pass and second-pass datasets, change only `DATASET_DIR`:

    DATASET_DIR = "../dataset"
    # or
    DATASET_DIR = "../dataset_refined"

Split ratios
------------
The default proportions are:
- training   : 70%
- validation : 15%
- test       : 15%

Side effects
------------
- Deletes any existing `../dataset_split` directory before creating a new split.
- Copies images into the new split folders using `shutil.copy`.
- Leaves the original dataset unchanged.
"""

import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


# Source dataset directory.
# Change this line only when switching between the first and second pass.
DATASET_DIR = "../dataset"
# DATASET_DIR = "../dataset_refined"

# Output directory for the generated train/validation/test split.
OUTPUT_DIR = "../dataset_split"

# Target split proportions.
SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

# Supported grade labels.
GRADES = ["A", "B", "C"]


def collect_images_by_grade(root: Path):
    """
    Collect image paths grouped by grade label.

    The function scans the source dataset directory and returns a dictionary mapping
    each grade to the list of image paths assigned to that grade.

    Supported source folder formats
    -------------------------------
    1. Fruit__Grade
       Example: Apple__A

    2. Grade-only
       Example: A

    Parameters
    ----------
    root : pathlib.Path
        Path to the source dataset directory.

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Dictionary of the form:
            {
                "A": [Path(...), Path(...)],
                "B": [Path(...), ...],
                "C": [Path(...), ...]
            }

    Notes
    -----
    - Only directories are processed.
    - Only grades listed in `GRADES` are accepted.
    - All files inside a valid grade folder are included.
    - Subdirectories inside grade folders are ignored.
    """
    grade_to_images = {g: [] for g in GRADES}

    for entry in os.listdir(root):
        entry_path = root / entry
        if not entry_path.is_dir():
            continue

        # Case 1: folder name follows the pattern "<fruit>__<grade>".
        if "__" in entry:
            _, grade = entry.split("__", 1)

        # Case 2: folder name is simply the grade label.
        else:
            grade = entry

        grade = grade.strip()
        if grade not in GRADES:
            continue

        for f in os.listdir(entry_path):
            p = entry_path / f
            if p.is_file():
                grade_to_images[grade].append(p)

    return grade_to_images


def main():
    """
    Run the dataset splitting pipeline.

    Workflow
    --------
    1. Remove any existing output split directory.
    2. Recreate the split folder structure.
    3. Group source images by grade.
    4. Split each grade independently into train, validation, and test subsets.
    5. Copy images into the corresponding output folders.

    Splitting logic
    ---------------
    For each grade:
    - First split into train and temporary subsets.
    - Then split the temporary subset into validation and test subsets.
    - A fixed random seed is used for reproducibility.

    Returns
    -------
    None

    Notes
    -----
    - Images are copied, not moved.
    - Class-wise splitting helps preserve grade distribution across the three subsets.
    - Very small classes may cause `train_test_split` to fail if there are not enough
      images to satisfy the requested proportions.
    """
    root = Path(DATASET_DIR)

    # Remove any previous split output so a fresh split is created each time.
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)

    # Create the root output directory and the top-level split folders.
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    for split in SPLIT:
        Path(OUTPUT_DIR, split).mkdir(exist_ok=True)

    # Collect source images grouped by grade label.
    grade_to_images = collect_images_by_grade(root)

    # Split each grade independently to preserve class structure.
    for grade, images in grade_to_images.items():
        if len(images) == 0:
            print(f"Skipping empty grade: {grade}")
            continue

        # First split: training set and temporary holdout set.
        train_imgs, temp = train_test_split(
            images,
            test_size=1 - SPLIT["train"],
            random_state=42,
        )

        # Second split: divide the temporary holdout into validation and test sets.
        val_imgs, test_imgs = train_test_split(
            temp,
            test_size=SPLIT["test"] / (SPLIT["test"] + SPLIT["val"]),
            random_state=42,
        )

        # Copy each subset into its corresponding output folder.
        for split_name, split_imgs in zip(
            ["train", "val", "test"],
            [train_imgs, val_imgs, test_imgs],
        ):
            out_dir = Path(OUTPUT_DIR) / split_name / grade
            out_dir.mkdir(parents=True, exist_ok=True)

            for img in split_imgs:
                shutil.copy(img, out_dir / img.name)

    print("Dataset split complete.")


if __name__ == "__main__":
    main()