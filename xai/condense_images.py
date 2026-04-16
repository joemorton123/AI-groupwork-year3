import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "../classifier/model_AtoC.pth"
TEST_DIR = "../dataset_split/test"
TOP_ERRORS_CSV = "../evaluation/results/top_errors.csv"
DATASET_DIR = "../dataset"

XAI_DIR = Path("xai")
(XAI_DIR / "gradcam").mkdir(parents=True, exist_ok=True)
(XAI_DIR / "integrated_gradients").mkdir(exist_ok=True)
(XAI_DIR / "shap").mkdir(exist_ok=True)
(XAI_DIR / "lime").mkdir(exist_ok=True)
(XAI_DIR / "robustness").mkdir(exist_ok=True)
(XAI_DIR / "top_errors").mkdir(exist_ok=True)

GRADES = ["A", "B", "C"]

def make_gradcam_grid():
    folder = XAI_DIR / "gradcam_per_fruit"
    out_path = folder / "gradcam_per_fruit_grid.jpg"

    # Collect images
    fruit_dict = {}

    for img_path in folder.glob("*.jpg"):
        name = img_path.stem
        if "_" not in name:
            continue

        fruit, state = name.split("_")
        fruit = fruit.capitalize()
        state = state.lower()

        if fruit not in fruit_dict:
            fruit_dict[fruit] = {"healthy": None, "rotten": None}

        if state in fruit_dict[fruit]:
            fruit_dict[fruit][state] = img_path

    # Filter fruits that have both images
    fruits = [f for f in fruit_dict if fruit_dict[f]["healthy"] and fruit_dict[f]["rotten"]]
    fruits.sort()

    if not fruits:
        print("[GRID] No complete fruit pairs found.")
        return

    # Create grid: 1 row per fruit, 2 columns (healthy | rotten)
    rows = len(fruits)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))

    if rows == 1:
        axes = [axes]  # ensure iterable

    for i, fruit in enumerate(fruits):
        healthy_path = fruit_dict[fruit]["healthy"]
        rotten_path = fruit_dict[fruit]["rotten"]

        healthy_img = Image.open(healthy_path)
        rotten_img = Image.open(rotten_path)

        # Left column: healthy
        axes[i][0].imshow(healthy_img)
        axes[i][0].set_title(f"{fruit} — Healthy")
        axes[i][0].axis("off")

        # Right column: rotten
        axes[i][1].imshow(rotten_img)
        axes[i][1].set_title(f"{fruit} — Rotten")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[GRID] Saved combined grid → {out_path}")

def make_top_errors_grids():
    folder = XAI_DIR / "top_errors"

    # Find all error folders
    error_folders = [f for f in folder.iterdir() if f.is_dir()]
    error_folders.sort()

    if not error_folders:
        print("[GRID] No top error folders found.")
        return

    for err_dir in error_folders:
        # Expected files
        gradcam_path = err_dir / "gradcam.jpg"
        ig_path = err_dir / "ig.jpg"
        shap_path = err_dir / "shap.jpg"
        lime_path = err_dir / "lime.jpg"

        # Skip if any missing
        if not (gradcam_path.exists() and ig_path.exists() and shap_path.exists() and lime_path.exists()):
            print(f"[GRID] Skipping {err_dir.name} — missing one or more XAI images.")
            continue

        # Load images
        gradcam_img = Image.open(gradcam_path)
        ig_img = Image.open(ig_path)
        shap_img = Image.open(shap_path)
        lime_img = Image.open(lime_path)

        # Create 1×4 grid
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(gradcam_img)
        axes[0].set_title("Grad-CAM")
        axes[0].axis("off")

        axes[1].imshow(ig_img)
        axes[1].set_title("Integrated Gradients")
        axes[1].axis("off")

        axes[2].imshow(shap_img)
        axes[2].set_title("SHAP")
        axes[2].axis("off")

        axes[3].imshow(lime_img)
        axes[3].set_title("LIME")
        axes[3].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save grid
        out_path = err_dir / f"{err_dir.name}_grid.jpg"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[GRID] Saved → {out_path}")
    
def main():
    # Condense image folders into 1 image
    print("[XAI] Creating Grad-CAM grid of one healthy + one rotten per fruit...")
    make_gradcam_grid()

    print("[XAI] Creating top error grids...")
    make_top_errors_grids()

if __name__ == "__main__":
    main()