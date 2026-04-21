"""
Visualise the highest-confidence misclassifications from model evaluation.

Purpose
-------
This script reads the `top_errors.csv` file produced during model evaluation,
locates the corresponding original images in the dataset, annotates each image with
its predicted label, true label, and confidence score, and saves both:

1. Individual annotated error images
2. A combined grid image showing all selected top errors

Generated outputs
-----------------
- results/top_error_images/<filename>_annotated.jpg
    Individually annotated versions of each top-error image

- results/top_errors_visualised.png
    A tiled grid containing all annotated top-error images

Assumptions
-----------
- `results/top_errors.csv` already exists and contains the columns:
    - filename
    - predicted
    - true
    - confidence
- The original dataset still exists at `../dataset`.
- Filenames are sufficient to locate images uniquely in the original dataset.
- Pillow can open and process the matching image files.

Side effects
------------
- Creates `results/top_error_images/` if it does not already exist.
- Writes annotated image files and a combined grid image to disk.
"""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Directory containing evaluation outputs.
RESULTS_DIR = Path("results")

# CSV file listing the highest-confidence misclassifications.
TOP_ERRORS_CSV = RESULTS_DIR / "top_errors.csv"

# Original dataset used to recover the full image files.
ORIGINAL_DATASET = Path("../dataset")

# Output directory for individual annotated misclassification images.
OUTPUT_DIR = RESULTS_DIR / "top_error_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed size used when displaying and composing images in the final grid.
DISPLAY_SIZE = (300, 300)


def find_original_image(filename):
    """
    Search the original dataset for a file with the given filename.

    Parameters
    ----------
    filename : str
        Name of the image file to locate.

    Returns
    -------
    pathlib.Path or None
        Full path to the matching image if found; otherwise `None`.

    Notes
    -----
    The search scans only the first level of subfolders inside `ORIGINAL_DATASET`.
    If duplicate filenames exist across folders, the first match found is returned.
    """
    for folder in ORIGINAL_DATASET.iterdir():
        if folder.is_dir():
            candidate = folder / filename
            if candidate.exists():
                return candidate
    return None


def annotate_image(img, predicted, true, confidence):
    """
    Add a text overlay showing predicted label, true label, and confidence.

    Parameters
    ----------
    img : PIL.Image.Image
        Image to annotate.
    predicted : str
        Predicted class label.
    true : str
        Ground-truth class label.
    confidence : str
        Confidence score, typically read from the CSV file.

    Returns
    -------
    PIL.Image.Image
        A copy of the input image with a text overlay added.

    Notes
    -----
    A solid dark rectangle is drawn behind the text to improve readability.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    text = f"Pred: {predicted}\nTrue: {true}\nConf: {confidence}"

    # Draw a background box for the annotation text.
    draw.rectangle([(0, 0), (150, 60)], fill=(0, 0, 0, 180))

    # Draw the annotation text in the top-left corner.
    draw.text((5, 5), text, fill="white")

    return img


def main():
    """
    Create annotated visualisations of top model errors.

    Workflow
    --------
    1. Load the top misclassification records from `top_errors.csv`.
    2. Locate each original image in the dataset.
    3. Resize each image to a fixed display size.
    4. Annotate each image with prediction details.
    5. Save each annotated image individually.
    6. Combine all annotated images into a single grid image.

    Returns
    -------
    None

    Notes
    -----
    - Entries whose original image cannot be found are skipped.
    - The final grid uses a fixed 5-column layout.
    - Empty CSV input results in early termination.
    """
    print("\n=== Visualising Top Errors ===")

    errors = []

    # Read the top error records from CSV.
    with open(TOP_ERRORS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            errors.append(row)

    if not errors:
        print("[ERROR] No errors found in CSV.")
        return

    print(f"[INFO] Found {len(errors)} top errors.")

    # Holds all successfully processed annotated images for the final grid.
    images = []

    for err in errors:
        filename = err["filename"]
        predicted = err["predicted"]
        true = err["true"]
        confidence = err["confidence"]

        print(f"[INFO] Processing {filename}...")

        original_path = find_original_image(filename)
        if original_path is None:
            print(f"[WARNING] Could not find original image for {filename}")
            continue

        # Load and standardise image size for display.
        img = Image.open(original_path).convert("RGB")
        img = img.resize(DISPLAY_SIZE)

        # Add text annotation.
        img = annotate_image(img, predicted, true, confidence)

        # Save the individual annotated image.
        img.save(OUTPUT_DIR / f"{filename}_annotated.jpg")

        images.append(img)

    # Create a tiled grid of all successfully processed images.
    if images:
        cols = 5
        rows = (len(images) + cols - 1) // cols

        grid_width = DISPLAY_SIZE[0] * cols
        grid_height = DISPLAY_SIZE[1] * rows

        grid = Image.new("RGB", (grid_width, grid_height), "white")

        for idx, img in enumerate(images):
            x = (idx % cols) * DISPLAY_SIZE[0]
            y = (idx // cols) * DISPLAY_SIZE[1]
            grid.paste(img, (x, y))

        grid.save(RESULTS_DIR / "top_errors_visualised.png")
        print("[SAVED] top_errors_visualised.png")

    print("\n=== Done. Check results/top_error_images/ and top_errors_visualised.png ===")


if __name__ == "__main__":
    main()