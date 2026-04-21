"""
Create composite visualisation grids for explainability outputs.

Purpose
-------
This script combines previously generated explainability images into larger summary
figures for easier inspection and reporting. It supports four types of composite
output:

1. Per-fruit Grad-CAM comparison grid
   - one row per fruit
   - two columns: healthy and rotten

2. Top-error explanation grids
   - one row per misclassified sample
   - four columns: Grad-CAM, Integrated Gradients, SHAP, and LIME

3. Robustness comparison grids
   - one row per source image
   - four columns: original, rotated, darker, and brighter

4. Method overview grids
   - one combined figure per XAI method
   - shows all available images for that method in a fixed grid layout

Expected directory structure
----------------------------
xai/
    gradcam/
    integrated_gradients/
    shap/
    lime/
    robustness/
    top_errors/
    gradcam_per_fruit/

Generated outputs
-----------------
- xai/gradcam_per_fruit/gradcam_per_fruit_grid.jpg
- xai/top_errors/<error_folder>/<error_folder>_grid.jpg
- xai/robustness/<base>_robustness_grid.jpg
- xai/<method>/<method>_all.jpg

Assumptions
-----------
- The required input images have already been generated and saved to the expected
  folders.
- Filenames follow the naming patterns expected by each grid-building function.
- Images are stored in `.jpg` format where required by the current logic.

Side effects
------------
- Creates the XAI output directory structure if it does not already exist.
- Writes combined grid images to disk.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from PIL import Image


# Root directory for explainability outputs.
XAI_DIR = Path("xai")

# Ensure the expected output subdirectories exist.
(XAI_DIR / "gradcam").mkdir(parents=True, exist_ok=True)
(XAI_DIR / "integrated_gradients").mkdir(exist_ok=True)
(XAI_DIR / "shap").mkdir(exist_ok=True)
(XAI_DIR / "lime").mkdir(exist_ok=True)
(XAI_DIR / "robustness").mkdir(exist_ok=True)
(XAI_DIR / "top_errors").mkdir(exist_ok=True)


def make_gradcam_grid():
    """
    Create a Grad-CAM comparison grid with one row per fruit.

    The function scans `xai/gradcam_per_fruit/` for images named in the format:

        <fruit>_healthy.jpg
        <fruit>_rotten.jpg

    Each fruit is included only if both healthy and rotten images are present.
    The final figure contains:
    - left column: healthy image
    - right column: rotten image

    Returns
    -------
    None

    Output
    ------
    xai/gradcam_per_fruit/gradcam_per_fruit_grid.jpg

    Notes
    -----
    - Fruits with incomplete healthy/rotten pairs are excluded.
    - Fruit names are capitalised for display.
    """
    folder = XAI_DIR / "gradcam_per_fruit"
    out_path = folder / "gradcam_per_fruit_grid.jpg"

    # Map each fruit to its healthy and rotten image paths.
    fruit_dict = {}

    for img_path in folder.glob("*.jpg"):
        name = img_path.stem
        if "_" not in name:
            continue

        parts = name.split("_")
        if len(parts) != 2:
            continue

        fruit, state = parts
        fruit = fruit.capitalize()
        state = state.lower()

        if fruit not in fruit_dict:
            fruit_dict[fruit] = {"healthy": None, "rotten": None}

        if state in fruit_dict[fruit]:
            fruit_dict[fruit][state] = img_path

    # Keep only fruits for which both healthy and rotten images are available.
    fruits = [f for f in fruit_dict if fruit_dict[f]["healthy"] and fruit_dict[f]["rotten"]]
    fruits.sort()

    if not fruits:
        print("[GRID] No complete fruit pairs found.")
        return

    rows = len(fruits)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 4 * rows))

    # When only one row exists, convert axes into a list-like structure.
    if rows == 1:
        axes = [axes]

    for i, fruit in enumerate(fruits):
        healthy_path = fruit_dict[fruit]["healthy"]
        rotten_path = fruit_dict[fruit]["rotten"]

        healthy_img = Image.open(healthy_path)
        rotten_img = Image.open(rotten_path)

        axes[i][0].imshow(healthy_img)
        axes[i][0].set_title(f"{fruit} — Healthy")
        axes[i][0].axis("off")

        axes[i][1].imshow(rotten_img)
        axes[i][1].set_title(f"{fruit} — Rotten")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[GRID] Saved combined grid → {out_path}")


def make_top_errors_grids():
    """
    Create one 1×4 explanation grid for each top-error folder.

    The function scans subdirectories inside `xai/top_errors/`. Each error folder is
    expected to contain the following files:

        gradcam.jpg
        ig.jpg
        shap.jpg
        lime.jpg

    A grid is generated only when all four images are present.

    Returns
    -------
    None

    Output
    ------
    xai/top_errors/<error_folder>/<error_folder>_grid.jpg

    Notes
    -----
    - Incomplete error folders are skipped.
    - Each grid contains four columns:
      Grad-CAM, Integrated Gradients, SHAP, and LIME.
    """
    folder = XAI_DIR / "top_errors"

    error_folders = [f for f in folder.iterdir() if f.is_dir()]
    error_folders.sort()

    if not error_folders:
        print("[GRID] No top error folders found.")
        return

    for err_dir in error_folders:
        gradcam_path = err_dir / "gradcam.jpg"
        ig_path = err_dir / "ig.jpg"
        shap_path = err_dir / "shap.jpg"
        lime_path = err_dir / "lime.jpg"

        if not (gradcam_path.exists() and ig_path.exists() and shap_path.exists() and lime_path.exists()):
            print(f"[GRID] Skipping {err_dir.name} — missing one or more XAI images.")
            continue

        gradcam_img = Image.open(gradcam_path)
        ig_img = Image.open(ig_path)
        shap_img = Image.open(shap_path)
        lime_img = Image.open(lime_path)

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

        out_path = err_dir / f"{err_dir.name}_grid.jpg"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[GRID] Saved → {out_path}")


def make_robustness_grids():
    """
    Create robustness comparison grids from Grad-CAM images.

    The function scans `xai/robustness/` for files named using the patterns:

        <base>_original_gradcam.jpg
        <base>_rotated_gradcam.jpg
        <base>_darker_gradcam.jpg
        <base>_brighter_gradcam.jpg

    A grid is created only when all four variants exist for the same base name.

    Returns
    -------
    None

    Output
    ------
    xai/robustness/<base>_robustness_grid.jpg

    Notes
    -----
    Each grid contains four columns:
    - Original
    - Rotated
    - Darker
    - Brighter
    """
    folder = XAI_DIR / "robustness"

    images = list(folder.glob("*_gradcam.jpg"))
    if not images:
        print("[GRID] No robustness images found.")
        return

    # Group the four robustness variants under a shared base name.
    groups = {}

    for img_path in images:
        name = img_path.stem
        if "_gradcam" not in name:
            continue

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
            groups[base] = {
                "original": None,
                "rotated": None,
                "darker": None,
                "brighter": None,
            }

        groups[base][variant] = img_path

    for base, variants in groups.items():
        if not all(variants.values()):
            print(f"[GRID] Skipping {base} — missing one or more robustness images.")
            continue

        imgs = [
            Image.open(variants["original"]),
            Image.open(variants["rotated"]),
            Image.open(variants["darker"]),
            Image.open(variants["brighter"]),
        ]
        titles = ["Original", "Rotated", "Darker", "Brighter"]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        plt.subplots_adjust(top=0.88)

        for i in range(4):
            axes[i].imshow(imgs[i])
            axes[i].set_title(titles[i])
            axes[i].axis("off")

        out_path = folder / f"{base}_robustness_grid.jpg"
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[GRID] Saved → {out_path}")


def make_method_overview_grid(method_name):
    """
    Create a combined overview grid for one explainability method.

    The function scans `xai/<method_name>/` for `.jpg` files and combines them into a
    fixed 2×3 grid. The filename stem of each image is wrapped and displayed as the
    subplot title.

    Parameters
    ----------
    method_name : str
        Name of the XAI method subdirectory. Expected values include:
        - "gradcam"
        - "integrated_gradients"
        - "shap"
        - "lime"

    Returns
    -------
    None

    Output
    ------
    xai/<method_name>/<method_name>_all.jpg

    Notes
    -----
    - The current layout is fixed at 2 rows × 3 columns.
    - Empty grid cells remain blank if fewer than six images are available.
    - Additional images beyond six will not be shown because only six subplot
      positions are created.
    """
    folder = XAI_DIR / method_name
    out_path = folder / f"{method_name}_all.jpg"

    images = sorted(folder.glob("*.jpg"))
    if not images:
        print(f"[GRID] No images found in {folder}")
        return

    pil_images = [Image.open(p) for p in images]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    plt.subplots_adjust(top=0.92, hspace=0.4)

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            ax.imshow(pil_images[i])
            wrapped = "\n".join(textwrap.wrap(images[i].stem, width=35))
            ax.set_title(wrapped, fontsize=9)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[GRID] Saved → {out_path}")


def main():
    """
    Run all XAI grid-generation steps.

    Workflow
    --------
    1. Create the per-fruit Grad-CAM comparison grid.
    2. Create per-error explanation grids for top misclassifications.
    3. Create robustness comparison grids.
    4. Create overview grids for each explanation method.

    Returns
    -------
    None
    """
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