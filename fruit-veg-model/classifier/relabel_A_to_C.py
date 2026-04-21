import os
from pathlib import Path
import cv2

DATASET_DIR = "../uncertain"
VALID_GRADES = ["A", "B", "C"]

def iter_images(root):
    root = Path(root)
    return [p for p in root.iterdir() if p.is_file()]

def main():
    print("Relabelling tool:")
    print("  A/B/C = assign grade")
    print("  X = delete image")
    print("  N = skip")
    print("  Q = quit")

    images = iter_images(DATASET_DIR)
    total = len(images)

    for idx, path in enumerate(images, start=1):
        img = cv2.imread(str(path))
        if img is None:
            print(f"[{idx}/{total}] Skipping unreadable image: {path}")
            continue

        # Show progress in window title
        window_title = f"[{idx}/{total}] {path.name}"
        disp = img.copy()
        cv2.putText(disp, path.name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)

        # Quit
        if key in [ord("q"), ord("Q")]:
            break

        # Skip
        if key in [ord("n"), ord("N")]:
            print(f"[{idx}/{total}] Skipped {path.name}")
            continue

        # Delete
        if key in [ord("x"), ord("X")]:
            print(f"[{idx}/{total}] Deleted {path.name}")
            path.unlink()
            continue

        # Assign grade
        new_grade = chr(key).upper()
        if new_grade not in VALID_GRADES:
            print(f"[{idx}/{total}] Invalid key, skipping")
            continue

        refined_dir = Path("../dataset_refined") / new_grade
        refined_dir.mkdir(parents=True, exist_ok=True)

        dest = refined_dir / path.name
        os.replace(path, dest)

        print(f"[{idx}/{total}] Moved {path.name} → {new_grade}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()