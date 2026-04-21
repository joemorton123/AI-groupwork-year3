"""
Interactive relabelling tool for uncertain fruit images.

Purpose
-------
This script reviews images stored in an "uncertain" directory and allows manual
reassignment into quality grades using keyboard input. Each image is displayed
one at a time in an OpenCV window, and an action is selected from the keyboard:

- A / B / C : move the image into the corresponding grade folder
- X         : delete the image
- N         : skip the image and leave it unchanged
- Q         : quit the tool immediately

Destination structure
---------------------
Labelled images are moved into:

    ../dataset_refined/A
    ../dataset_refined/B
    ../dataset_refined/C

Assumptions
-----------
- The source directory exists at `../uncertain`.
- Images are stored directly inside that directory.
- OpenCV can open a display window in the current environment.
- Folder names are grade-only, not fruit-specific.

Side effects
------------
- Creates destination grade folders if they do not already exist.
- Moves labelled images on disk using `os.replace`.
- Permanently deletes images labelled with `X`.
"""

import os
from pathlib import Path

import cv2


# Directory containing images that require manual review.
DATASET_DIR = "../uncertain"

# Allowed quality grades for manual assignment.
VALID_GRADES = ["A", "B", "C"]


def iter_images(root):
    """
    Return all files located directly inside the given directory.

    Parameters
    ----------
    root : str or Path
        Path to the directory containing uncertain images.

    Returns
    -------
    list[pathlib.Path]
        A list of file paths found in the directory.

    Notes
    -----
    - Only direct child files are included.
    - Subdirectories are ignored.
    - No filtering by file extension is applied here.
    """
    root = Path(root)
    return [p for p in root.iterdir() if p.is_file()]


def main():
    """
    Run the interactive image relabelling workflow.

    Workflow
    --------
    1. Load all files from the uncertain-image directory.
    2. Display each readable image in an OpenCV window.
    3. Wait for a keyboard action:
       - A/B/C: assign a grade and move the image
       - X: delete the image
       - N: skip the image
       - Q: stop processing
    4. Close any remaining OpenCV windows before exit.

    Returns
    -------
    None

    Notes
    -----
    - Unreadable images are skipped automatically.
    - `os.replace` may overwrite an existing file if the same filename already
      exists in the destination folder.
    - `path.unlink()` permanently removes the file from disk.
    """
    print("Relabelling tool:")
    print("  A/B/C = assign grade")
    print("  X = delete image")
    print("  N = skip")
    print("  Q = quit")

    images = iter_images(DATASET_DIR)
    total = len(images)

    for idx, path in enumerate(images, start=1):
        img = cv2.imread(str(path))

        # Skip files that cannot be opened as images.
        if img is None:
            print(f"[{idx}/{total}] Skipping unreadable image: {path}")
            continue

        # Show progress and filename in the window title.
        window_title = f"[{idx}/{total}] {path.name}"

        # Create a copy so the original image array remains unchanged.
        disp = img.copy()

        # Overlay the filename on the displayed image for reference.
        cv2.putText(
            disp,
            path.name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)

        # Quit the tool immediately.
        if key in [ord("q"), ord("Q")]:
            break

        # Leave the image unchanged and continue to the next one.
        if key in [ord("n"), ord("N")]:
            print(f"[{idx}/{total}] Skipped {path.name}")
            continue

        # Delete the image permanently.
        if key in [ord("x"), ord("X")]:
            print(f"[{idx}/{total}] Deleted {path.name}")
            path.unlink()
            continue

        # Convert the pressed key to an uppercase grade label.
        new_grade = chr(key).upper()

        # Ignore unsupported keys.
        if new_grade not in VALID_GRADES:
            print(f"[{idx}/{total}] Invalid key, skipping")
            continue

        # Create the destination grade folder if needed.
        refined_dir = Path("../dataset_refined") / new_grade
        refined_dir.mkdir(parents=True, exist_ok=True)

        # Move the file into the selected grade folder.
        dest = refined_dir / path.name
        os.replace(path, dest)

        print(f"[{idx}/{total}] Moved {path.name} → {new_grade}")

    # Ensure all OpenCV windows are closed before exit.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()