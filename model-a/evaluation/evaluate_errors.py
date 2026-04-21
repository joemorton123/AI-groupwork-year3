import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = Path("results")
TOP_ERRORS_CSV = RESULTS_DIR / "top_errors.csv"
ORIGINAL_DATASET = Path("../dataset")
OUTPUT_DIR = RESULTS_DIR / "top_error_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed display size
DISPLAY_SIZE = (300, 300)

def find_original_image(filename):
    """
    Search the original dataset for the given filename.
    Returns full path or None.
    """
    for folder in ORIGINAL_DATASET.iterdir():
        if folder.is_dir():
            candidate = folder / filename
            if candidate.exists():
                return candidate
    return None

def annotate_image(img, predicted, true, confidence):
    """
    Add text overlay with predicted/true/confidence.
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    text = f"Pred: {predicted}\nTrue: {true}\nConf: {confidence}"
    draw.rectangle([(0, 0), (150, 60)], fill=(0, 0, 0, 180))
    draw.text((5, 5), text, fill="white")

    return img

def main():
    print("\n=== Visualising Top Errors ===")

    errors = []
    with open(TOP_ERRORS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            errors.append(row)

    if not errors:
        print("[ERROR] No errors found in CSV.")
        return

    print(f"[INFO] Found {len(errors)} top errors.")

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

        img = Image.open(original_path).convert("RGB")
        img = img.resize(DISPLAY_SIZE)

        img = annotate_image(img, predicted, true, confidence)

        # Save individual annotated image
        img.save(OUTPUT_DIR / f"{filename}_annotated.jpg")

        images.append(img)

    # Create a grid of all images
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