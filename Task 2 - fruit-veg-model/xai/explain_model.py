"""
Explainability pipeline for a three-class fruit-grade classifier.

Purpose
-------
This script generates multiple explainable AI (XAI) outputs for a trained
EfficientNet-B0 fruit-grading model. The supported explanation methods are:

- Grad-CAM
- Integrated Gradients
- GradientSHAP
- LIME

The script also produces:
- Grad-CAM examples for one healthy and one rotten sample per fruit
- Grad-CAM robustness views for simple image perturbations
- XAI explanations for selected top-error cases from model evaluation

Expected inputs
---------------
- Trained model checkpoint:
    ../classifier/model_AtoC.pth

- Test dataset:
    ../dataset_split/test

- Top error CSV from evaluation:
    ../evaluation/results/top_errors.csv

- Original dataset:
    ../dataset

Generated outputs
-----------------
Outputs are written under the `xai/` directory:

xai/
    gradcam/
    integrated_gradients/
    shap/
    lime/
    robustness/
    top_errors/
    gradcam_per_fruit/

Assumptions
-----------
- The trained model predicts exactly three classes: A, B, and C.
- The test dataset and original dataset exist at the configured paths.
- Top-error filenames are sufficient to recover the corresponding source images.
- The runtime environment supports all required dependencies:
  PyTorch, torchvision, Captum, LIME, OpenCV, matplotlib, scikit-image, and Pillow.

Side effects
------------
- Existing XAI output folders are deleted and recreated at the start of execution.
- Multiple image files are written to disk.
"""

import csv
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import GradientShap, IntegratedGradients
from lime import lime_image
from PIL import Image
from skimage import segmentation
from torchvision import models, transforms


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


def load_model():
    """
    Load the trained EfficientNet-B0 classifier.

    The saved checkpoint is loaded into an EfficientNet-B0 architecture whose
    classifier head has been replaced with a three-class linear layer.

    Returns
    -------
    torch.nn.Module
        The loaded model in evaluation mode.

    Notes
    -----
    - Weights are loaded onto CPU using `map_location="cpu"`.
    - `model.eval()` is used to disable training-specific behaviour.
    """
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# Load the model once so that all explanation methods can reuse it.
model = load_model()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_image(path):
    """
    Load an image from disk and prepare its model input tensor.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    tuple[PIL.Image.Image or None, torch.Tensor or None]
        A tuple containing:
        - the loaded PIL image in RGB format
        - the transformed tensor with an added batch dimension

        If loading fails, `(None, None)` is returned.

    Notes
    -----
    - The returned tensor has shape `(1, C, H, W)`.
    - Errors are handled by skipping unreadable images.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        print(f"[LOAD] SKIPPED — cannot load image: {path}")
        return None, None

    return img, transform(img).unsqueeze(0)


def tensor_from_pil(img):
    """
    Convert a PIL image into a batched model input tensor.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(1, C, H, W)`.
    """
    return transform(img).unsqueeze(0)


# -----------------------------
# Grad-CAM
# -----------------------------
def gradcam(img_tensor, img_pil, save_path):
    """
    Generate and save a Grad-CAM overlay for a single image.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Input image tensor with shape `(1, C, H, W)`.
    img_pil : PIL.Image.Image
        Original image used for visual overlay.
    save_path : Path
        Destination path for the saved Grad-CAM image.

    Returns
    -------
    None

    Method
    ------
    1. Detect the last convolutional layer in the model.
    2. Register forward and backward hooks.
    3. Run a forward pass and identify the predicted class.
    4. Backpropagate from the predicted class score.
    5. Compute channel weights using global average pooling of gradients.
    6. Form the class activation map from weighted activations.
    7. Resize, colourise, and overlay the heatmap on the original image.

    Notes
    -----
    - The last `Conv2d` layer is selected automatically.
    - Only the predicted class is explained.
    """
    print(f"[GradCAM] Processing {save_path.name}")

    model.zero_grad()

    # Automatically select the final convolutional layer.
    target_layer = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print("[GradCAM] ERROR: No Conv2d layer found")
        return

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        """Store forward activations from the target convolutional layer."""
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        """Store gradients flowing out of the target convolutional layer."""
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    model.zero_grad()
    logits[0, pred_class].backward()

    fh.remove()
    bh.remove()

    act = activations[0].detach().squeeze(0)
    grad = gradients[0].detach().squeeze(0)

    if act.ndim != 3:
        print(f"[GradCAM] SKIPPED — invalid activation shape: {act.shape}")
        return

    if grad.ndim != 3:
        print(f"[GradCAM] SKIPPED — invalid gradient shape: {grad.shape}")
        return

    # Global average pooling over the spatial dimensions.
    weights = grad.mean(dim=(1, 2))

    # Weighted sum of feature maps.
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = cam.cpu().numpy()
    cam = np.maximum(cam, 0)

    # Normalise the activation map to [0, 1].
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_resized = cv2.resize(cam, (224, 224))
    cam_uint8 = np.uint8(cam_resized * 255)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    img_np = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(str(save_path), overlay[:, :, ::-1])
    print(f"[GradCAM] Saved → {save_path}")


def gradcam_one_per_fruit():
    """
    Generate one healthy and one rotten Grad-CAM example per fruit.

    Folder naming is expected to follow the pattern:

        <fruit>__<grade>

    Grade interpretation:
    - A -> healthy
    - B/C -> rotten

    Returns
    -------
    None

    Output
    ------
    xai/gradcam_per_fruit/<Fruit>_healthy.jpg
    xai/gradcam_per_fruit/<Fruit>_rotten.jpg

    Notes
    -----
    Only the first valid image found for each fruit/state combination is used.
    """
    print("[XAI] Generating 1 healthy + 1 rotten Grad-CAM per fruit...")

    fruit_samples = {}

    for folder in Path(DATASET_DIR).iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        if "__" not in folder_name:
            continue

        fruit, grade = folder_name.split("__")
        fruit = fruit.capitalize()
        grade = grade.upper()

        state = "healthy" if grade == "A" else "rotten"

        if fruit not in fruit_samples:
            fruit_samples[fruit] = {"healthy": None, "rotten": None}

        if fruit_samples[fruit][state] is None:
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    fruit_samples[fruit][state] = img_path
                    break

    out_dir = XAI_DIR / "gradcam_per_fruit"
    out_dir.mkdir(exist_ok=True)

    for fruit, samples in fruit_samples.items():
        for state, path in samples.items():
            if path is None:
                continue

            img, tensor = load_image(path)
            if img is None:
                continue

            save_path = out_dir / f"{fruit}_{state}.jpg"
            print(f"[XAI] Grad-CAM for {fruit} ({state}) → {save_path.name}")
            gradcam(tensor, img, save_path)

    print("[XAI] Completed per-fruit Grad-CAM.")


# -----------------------------
# Integrated Gradients
# -----------------------------
def integrated_gradients(img_tensor, img_pil, save_path):
    """
    Generate and save an Integrated Gradients overlay.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Input image tensor of shape `(1, C, H, W)`.
    img_pil : PIL.Image.Image
        Original image for visual overlay.
    save_path : Path
        Destination path for the saved image.

    Returns
    -------
    None

    Method
    ------
    - A black image is used as the baseline.
    - Attributions are computed for the predicted class.
    - Channel attributions are averaged into a single grayscale map.
    - Absolute values are used to highlight attribution magnitude.
    - The resulting heatmap is overlaid on the original image.
    """
    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(img_tensor)

    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    attributions = ig.attribute(img_tensor, baseline, target=pred_class)
    attributions = attributions.squeeze(0).cpu().numpy()
    attributions = np.mean(attributions, axis=0)
    attributions = np.abs(attributions)

    attributions = (
        (attributions - attributions.min()) /
        (attributions.max() - attributions.min() + 1e-8)
    )

    attr_uint8 = np.uint8(attributions * 255)
    heatmap = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)

    img_np = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(str(save_path), overlay[:, :, ::-1])
    print(f"[IG] Saved → {save_path}")


# -----------------------------
# SHAP
# -----------------------------
def shap_explain(img_tensor, img_pil, save_path):
    """
    Generate and save a GradientSHAP overlay.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Input image tensor of shape `(1, C, H, W)`.
    img_pil : PIL.Image.Image
        Original image for visual overlay.
    save_path : Path
        Destination path for the saved image.

    Returns
    -------
    None

    Method
    ------
    - A small random background set is sampled from the training split.
    - GradientSHAP is run for the predicted class.
    - Channel attributions are averaged into a grayscale map.
    - Absolute magnitude, contrast stretching, and Gaussian smoothing are applied.
    - The heatmap is blended with the original image.

    Notes
    -----
    The current background set uses up to 10 random training images.
    """
    model.eval()

    train_images = list(Path("../dataset_split/train").rglob("*.jpg"))
    random.shuffle(train_images)

    background_imgs = []
    for p in train_images[:10]:
        _, t = load_image(p)
        if t is not None:
            background_imgs.append(t.squeeze(0))

    if not background_imgs:
        print("[SHAP] SKIPPED — no valid background images found")
        return

    background = torch.stack(background_imgs).to(img_tensor.device)

    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    explainer = GradientShap(model)

    attributions = explainer.attribute(
        img_tensor,
        baselines=background,
        target=pred_class,
        n_samples=50,
        stdevs=0.1,
    )

    attr = attributions.squeeze().cpu().detach().numpy()
    attr_gray = np.mean(attr, axis=0)
    attr_gray = np.abs(attr_gray)

    p1, p99 = np.percentile(attr_gray, (1, 99))
    attr_gray = np.clip((attr_gray - p1) / (p99 - p1 + 1e-8), 0, 1)

    attr_gray = cv2.GaussianBlur(attr_gray, (11, 11), sigmaX=5)

    attr_uint8 = np.uint8(attr_gray * 255)
    heatmap = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] Saved → {save_path}")


# -----------------------------
# LIME
# -----------------------------
def lime_explain(img_pil, save_path):
    """
    Generate and save a LIME explanation overlay.

    Parameters
    ----------
    img_pil : PIL.Image.Image
        Input image as a PIL object.
    save_path : Path
        Destination path for the saved explanation image.

    Returns
    -------
    None

    Method
    ------
    - LIME perturbs superpixels derived from SLIC segmentation.
    - Class probabilities are produced by the model for perturbed samples.
    - The explanation mask is converted into a smoothed heatmap.
    - The heatmap is blended with the original image.
    """
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        """
        Convert a batch of NumPy images into model probabilities.

        Parameters
        ----------
        images : list[numpy.ndarray]
            List of image arrays.

        Returns
        -------
        numpy.ndarray
            Model class probabilities for each image.
        """
        tensors = torch.stack([transform(Image.fromarray(img)) for img in images])
        logits = model(tensors)
        probs = F.softmax(logits, dim=1)
        return probs.detach().numpy()

    explanation = explainer.explain_instance(
        np.array(img_pil.resize((224, 224))),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=lambda x: segmentation.slic(
            x,
            n_segments=50,
            compactness=10,
            sigma=1,
        ),
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        hide_rest=False,
        min_weight=0.01,
    )

    mask = mask.astype(float)
    mask = cv2.GaussianBlur(mask, (11, 11), sigmaX=5)

    heatmap = cv2.applyColorMap(
        np.uint8((mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay / 255.0, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"[LIME] Saved → {save_path}")


# -----------------------------
# Robustness XAI
# -----------------------------
def robustness_explain(img_path):
    """
    Generate Grad-CAM explanations for perturbed versions of one image.

    The following variants are created:
    - original
    - rotated
    - darker
    - brighter

    Parameters
    ----------
    img_path : str or Path
        Path to the source image.

    Returns
    -------
    None

    Output
    ------
    xai/robustness/<image_stem>_<variant>_gradcam.jpg

    Notes
    -----
    Each variant is converted into its own input tensor before Grad-CAM is run.
    """
    img, _ = load_image(img_path)
    if img is None:
        print(f"[ROBUSTNESS] SKIPPED — could not load image: {img_path}")
        return

    variants = {
        "original": img,
        "rotated": img.rotate(20),
        "darker": Image.fromarray((np.array(img) * 0.6).astype(np.uint8)),
        "brighter": Image.fromarray((np.array(img) * 1.4).clip(0, 255).astype(np.uint8)),
    }

    for name, variant in variants.items():
        variant_tensor = tensor_from_pil(variant)
        save_path = XAI_DIR / "robustness" / f"{Path(img_path).stem}_{name}_gradcam.jpg"
        gradcam(variant_tensor, variant, save_path)

    print(f"[ROBUSTNESS] Completed → {img_path}")


# -----------------------------
# Top error explanations
# -----------------------------
def explain_top_errors():
    """
    Generate XAI outputs for selected top-error cases.

    The first five entries from `TOP_ERRORS_CSV` are processed. For each error:
    - the original image is located in the dataset
    - a dedicated output folder is created
    - Grad-CAM, Integrated Gradients, SHAP, and LIME are saved

    Returns
    -------
    None

    Output
    ------
    xai/top_errors/<filename>_true<label>_pred<label>/
        gradcam.jpg
        ig.jpg
        shap.jpg
        lime.jpg
    """
    with open(TOP_ERRORS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[TOP ERRORS] Total top errors: {len(rows)}")

    rows = rows[:5]
    print(f"[TOP ERRORS] Processing {len(rows)} items")

    for i, row in enumerate(rows, start=1):
        fname = row["filename"]
        true_label = row.get("true", "unknown")
        pred_label = row.get("predicted", "unknown")

        print(
            f"[TOP ERRORS] ({i}/{len(rows)}) {fname} — "
            f"true: {true_label}, pred: {pred_label}"
        )

        img_path = None
        for folder in Path(DATASET_DIR).iterdir():
            if not folder.is_dir():
                continue

            candidate = folder / fname
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"[TOP ERRORS] WARNING: Could not find {fname} in dataset")
            continue

        img, tensor = load_image(img_path)
        if img is None:
            print(f"[TOP ERRORS] WARNING: Could not load {fname}")
            continue

        folder_name = f"{Path(fname).stem}_true{true_label}_pred{pred_label}"
        error_dir = XAI_DIR / "top_errors" / folder_name
        error_dir.mkdir(parents=True, exist_ok=True)

        gradcam(tensor, img, error_dir / "gradcam.jpg")
        integrated_gradients(tensor, img, error_dir / "ig.jpg")
        shap_explain(tensor, img, error_dir / "shap.jpg")
        lime_explain(img, error_dir / "lime.jpg")

    print("[TOP ERRORS] Completed")


def clear_xai_folders():
    """
    Delete and recreate all XAI output folders.

    Returns
    -------
    None

    Notes
    -----
    Existing contents are permanently removed before new outputs are generated.
    """
    for folder in [
        "gradcam",
        "integrated_gradients",
        "shap",
        "lime",
        "robustness",
        "top_errors",
        "gradcam_per_fruit",
    ]:
        path = XAI_DIR / folder
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def main():
    """
    Run the full XAI generation pipeline.

    Workflow
    --------
    1. Clear previous XAI outputs.
    2. Select up to five random test images.
    3. Generate Grad-CAM, Integrated Gradients, SHAP, LIME, and robustness views.
    4. Generate one healthy and one rotten Grad-CAM example per fruit.
    5. Generate explanation sets for selected top-error cases.

    Returns
    -------
    None
    """
    print("=== Running XAI ===")
    skip_counter = 0
    analysis_counter = 0

    clear_xai_folders()

    test_images = list(Path(TEST_DIR).rglob("*.*"))
    random.shuffle(test_images)
    test_images = test_images[:5]

    for img_path in test_images:
        print(f"[MAIN] Running XAI on: {img_path}")

        if not str(img_path).lower().endswith((".jpg", ".jpeg", ".png")):
            skip_counter += 1
            continue

        img, tensor = load_image(img_path)
        if img is None:
            skip_counter += 1
            continue

        analysis_counter += 1

        print("[XAI] Running Grad-CAM...")
        gradcam(tensor, img, XAI_DIR / "gradcam" / f"{Path(img_path).stem}_gradcam.jpg")

        print("[XAI] Running Integrated Gradients...")
        integrated_gradients(
            tensor,
            img,
            XAI_DIR / "integrated_gradients" / f"{Path(img_path).stem}_ig.jpg",
        )

        print("[XAI] Running SHAP...")
        shap_explain(
            tensor,
            img,
            XAI_DIR / "shap" / f"{Path(img_path).stem}_shap.jpg",
        )

        print("[XAI] Running LIME...")
        lime_explain(
            img,
            XAI_DIR / "lime" / f"{Path(img_path).stem}_lime.jpg",
        )

        print("[XAI] Running Robustness tests...")
        robustness_explain(img_path)

    print("[XAI] Running Grad-CAM one per fruit...")
    gradcam_one_per_fruit()

    print("[XAI] Explaining top errors...")
    explain_top_errors()

    print("=== XAI complete. Check the xai/ folder ===")
    print(f"=== Images skipped: {skip_counter} with {analysis_counter} analysed ===")


if __name__ == "__main__":
    main()