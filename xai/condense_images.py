import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
XAI_DIR = Path("xai")
(XAI_DIR / "gradcam").mkdir(parents=True, exist_ok=True)
(XAI_DIR / "integrated_gradients").mkdir(exist_ok=True)
(XAI_DIR / "shap").mkdir(exist_ok=True)
(XAI_DIR / "lime").mkdir(exist_ok=True)
(XAI_DIR / "robustness").mkdir(exist_ok=True)
(XAI_DIR / "top_errors").mkdir(exist_ok=True)

# -----------------------------
# Gradcam grid: 1 row per fruit, 2 columns (healthy and rotten)
# -----------------------------
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

# -----------------------------
# Top error grids: 1 image per error, 4 columns (Grad-CAM, IG, SHAP, LIME)
# -----------------------------
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

# -----------------------------
# Robustness grids: 1 image per fruit, 4 columns (original, rotated, darker, brighter)
# -----------------------------
def make_robustness_grids():
    folder = XAI_DIR / "robustness"

    # Find all robustness images
    images = list(folder.glob("*_gradcam.jpg"))
    if not images:
        print("[GRID] No robustness images found.")
        return

    # Group by base name
    groups = {}

    for img_path in images:
        name = img_path.stem
        if "_gradcam" not in name:
            continue

        # Extract variant (original, rotated, darker, brighter)
        if "_original_gradcam" in name:
            variant = "original"
            base = name.replace("_original_gradcam", "")
        elif "_rotated_gradcam" in name:
            variant = "rotated"
            base = name.replace("_rotated_gradcam", "")
        elif "_darker_gradcam" in name:
            variant = "darker"
            base = name.replace("_darker_gradcam", "")
        elif "_brighter_gradcam" in name:
            variant = "brighter"
            base = name.replace("_brighter_gradcam", "")
        else:
            continue

        if base not in groups:
            groups[base] = {"original": None, "rotated": None, "darker": None, "brighter": None}

        groups[base][variant] = img_path

    # Create a grid for each robustness group
    for base, variants in groups.items():
        # Skip incomplete sets
        if not all(variants.values()):
            print(f"[GRID] Skipping {base} — missing one or more robustness images.")
            continue

        # Load images
        imgs = [
            Image.open(variants["original"]),
            Image.open(variants["rotated"]),
            Image.open(variants["darker"]),
            Image.open(variants["brighter"]),
        ]

        titles = ["Original", "Rotated", "Darker", "Brighter"]

        # Create 1×4 grid
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        plt.subplots_adjust(top=0.88)

        for i in range(4):
            axes[i].imshow(imgs[i])
            axes[i].set_title(titles[i])
            axes[i].axis("off")

        # Save output
        out_path = folder / f"{base}_robustness_grid.jpg"
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[GRID] Saved → {out_path}")

# -----------------------------
# Method overview grid: 1 image per method, show all fruits in 1 grid
# -----------------------------
def make_method_overview_grid(method_name):
    folder = XAI_DIR / method_name
    out_path = folder / f"{method_name}_all.jpg"

    # Collect all images
    images = sorted(folder.glob("*.jpg"))
    if not images:
        print(f"[GRID] No images found in {folder}")
        return

    # Load images
    pil_images = [Image.open(p) for p in images]

    # Determine grid layout (2 rows × 3 columns for 5 images)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    plt.subplots_adjust(top=0.92)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            ax.imshow(pil_images[i])
            ax.set_title(images[i].stem, fontsize=10)
        ax.axis("off")

    # Save combined figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[GRID] Saved → {out_path}")

# -----------------------------
# Main function to create all grids
# -----------------------------
def main():
    # Condense image folders into 1 image
    print("[XAI] Creating Grad-CAM grid of one healthy + one rotten per fruit...")
    make_gradcam_grid()

    print("[XAI] Creating top error grids...")
    make_top_errors_grids()

    print("[XAI] Creating robustness grids...")
    make_robustness_grids()

    print("[XAI] Creating method overview grids...")
    for method in ["gradcam", "integrated_gradients", "shap", "lime"]:
        make_method_overview_grid(method)

if __name__ == "__main__":
    main()