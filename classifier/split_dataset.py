import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# CHANGE THIS LINE ONLY when switching between first and second pass:
# First pass:
# DATASET_DIR = "../dataset"
# Second pass:
DATASET_DIR = "../dataset_refined"

OUTPUT_DIR = "../dataset_split"

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

GRADES = ["A", "B", "C", "D", "E"]

def collect_images_by_grade(root: Path):
    """
    Returns a dict: { 'A': [img_paths...], 'B': [...], ... }
    Works for:
      - Fruit__Grade folders (e.g. Apple__A)
      - Grade-only folders (A, B, C, D, E)
    """
    grade_to_images = {g: [] for g in GRADES}

    for entry in os.listdir(root):
        entry_path = root / entry
        if not entry_path.is_dir():
            continue

        # Case 1: Fruit__Grade
        if "__" in entry:
            fruit, grade = entry.split("__", 1)
        # Case 2: Grade-only folder
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
    root = Path(DATASET_DIR)

    # Clean old split directory
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    for split in SPLIT:
        Path(OUTPUT_DIR, split).mkdir(exist_ok=True)

    grade_to_images = collect_images_by_grade(root)

    for grade, images in grade_to_images.items():
        if len(images) == 0:
            print(f"Skipping empty grade: {grade}")
            continue

        train_imgs, temp = train_test_split(
            images,
            test_size=1 - SPLIT["train"],
            random_state=42,
        )
        val_imgs, test_imgs = train_test_split(
            temp,
            test_size=SPLIT["test"] / (SPLIT["test"] + SPLIT["val"]),
            random_state=42,
        )

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