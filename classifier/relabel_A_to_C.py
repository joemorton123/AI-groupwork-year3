import os
from pathlib import Path
import cv2

DATASET_DIR = "../uncertain"
VALID_GRADES = ["A", "B", "C"]

def iter_images(root):
    root = Path(root)

    # Case 1: flat folder (uncertain/)
    images = [p for p in root.iterdir() if p.is_file()]
    if len(images) > 0:
        for p in images:
            yield "Unknown", "?", p
        return

    # Case 2: fruit__grade folders (dataset/)
    for folder in os.listdir(root):
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        if "__" not in folder:
            continue
        fruit, grade = folder.split("__", 1)
        for f in os.listdir(folder_path):
            p = folder_path / f
            if p.is_file():
                yield fruit, grade, p

def main():
    print("Relabelling tool: press A/B/C to change grade, N to skip, Q to quit.")

    for fruit, grade, path in iter_images(DATASET_DIR):
        img = cv2.imread(str(path))
        if img is None:
            continue

        disp = img.copy()
        cv2.putText(disp, f"{fruit}__{grade}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Relabel", disp)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), ord("Q")]:
            break
        if key in [ord("n"), ord("N")]:
            continue

        new_grade = chr(key).upper()
        if new_grade not in VALID_GRADES:
            continue

        refined_dir = Path("../dataset_refined") / new_grade
        refined_dir.mkdir(parents=True, exist_ok=True)

        dest = refined_dir / path.name
        os.replace(path, dest)

        print(f"Moved {path} → {dest}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()