import torch
import torch.nn.functional as F
from torchvision import models, transforms
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
import torch.nn.functional as F
from lime import lime_image
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import shutil
import random
import csv
from skimage import segmentation

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

# -----------------------------
# Load model
# -----------------------------
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
    except:
        print(f"[LOAD] SKIPPED — cannot load image: {path}")
        return None, None
    return img, transform(img).unsqueeze(0)

# -----------------------------
# Grad-CAM
# -----------------------------
def gradcam(img_tensor, img_pil, save_path):
    print(f"[GradCAM] Processing {save_path.name}")

    model.zero_grad()

    # Autodetect last conv layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print("[GradCAM] ERROR: No Conv2d layer found")
        return

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    model.zero_grad()
    logits[0, pred_class].backward()

    fh.remove()
    bh.remove()

    act = activations[0].detach().squeeze(0) # shape (C, H, W)
    grad = gradients[0].detach().squeeze(0) # shape (C, H, W)

    # Ensure shapes are correct
    if act.ndim != 3:
        print(f"[GradCAM] SKIPPED — invalid activation shape: {act.shape}")
        return

    if grad.ndim != 3:
        print(f"[GradCAM] SKIPPED — invalid gradient shape: {grad.shape}")
        return

    # Global average pooling
    weights = grad.mean(dim=(1, 2)) # shape (C)

    # Weighted sum
    cam = (weights[:, None, None] * act).sum(dim=0) # shape (H, W)

    cam = cam.cpu().numpy()
    cam = np.maximum(cam, 0)

    # Normalise
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize to 224x224
    cam_resized = cv2.resize(cam, (224, 224))

    # Convert to uint8 single-channel
    cam_uint8 = np.uint8(cam_resized * 255)

    #print("CAM DEBUG:", cam_uint8.dtype, cam_uint8.shape)

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Overlay
    img_np = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(str(save_path), overlay[:, :, ::-1])
    print(f"[GradCAM] Saved → {save_path}")

def gradcam_one_per_fruit():
    print("[XAI] Generating 1 healthy + 1 rotten Grad-CAM per fruit...")

    fruit_samples = {} # fruit → {"healthy": path, "rotten": path}

    for folder in Path(DATASET_DIR).iterdir():
        if not folder.is_dir():
            continue

        # Folder name looks like: Apple__A
        folder_name = folder.name
        if "__" not in folder_name:
            continue

        fruit, grade = folder_name.split("__")

        fruit = fruit.capitalize()

        # Map A/B/C → healthy/rotten
        grade = grade.upper()
        if grade == "A":
            state = "healthy"
        else:
            state = "rotten"

        # Ensure dictionary entry exists
        if fruit not in fruit_samples:
            fruit_samples[fruit] = {"healthy": None, "rotten": None}

        # Only store first example
        if fruit_samples[fruit][state] is None:
            # Find first valid image in this folder
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    fruit_samples[fruit][state] = img_path
                    break

    # Output folder
    out_dir = XAI_DIR / "gradcam_per_fruit"
    out_dir.mkdir(exist_ok=True)

    # Run Grad-CAM for each fruit
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
    ig = IntegratedGradients(model)

    # Baseline = black image
    baseline = torch.zeros_like(img_tensor)

    # Predict class
    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    # Compute attributions → shape (1, 3, 224, 224)
    attributions = ig.attribute(img_tensor, baseline, target=pred_class)

    # Remove batch dimension → (3, 224, 224)
    attributions = attributions.squeeze(0).cpu().numpy()

    # Aggregate across channels → (224, 224)
    attributions = np.mean(attributions, axis=0)

    # Absolute value (IG can be negative)
    attributions = np.abs(attributions)

    # Normalise to 0-1
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)

    # Convert to uint8
    attr_uint8 = np.uint8(attributions * 255)

    # Apply colormap
    heatmap = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)

    # Overlay on original image
    img_np = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    # Save
    cv2.imwrite(str(save_path), overlay[:, :, ::-1])
    print(f"[IG] Saved → {save_path}")

# -----------------------------
# SHAP
# -----------------------------

def shap_explain(img_tensor, img_pil, save_path):
    model.eval()

    # Build realistic background batch (10 random training images)
    train_images = list(Path("../dataset_split/train").rglob("*.jpg"))
    random.shuffle(train_images)
    background_imgs = []

    for p in train_images[:10]:
        img, t = load_image(p)
        if t is not None:
            background_imgs.append(t.squeeze(0))

    background = torch.stack(background_imgs).to(img_tensor.device)

    # Predict class
    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    # GradientSHAP explainer
    explainer = GradientShap(model)

    attributions = explainer.attribute(
        img_tensor,
        baselines=background,
        target=pred_class,
        n_samples=50,
        stdevs=0.1
    )

    # Convert to numpy
    attr = attributions.squeeze().cpu().detach().numpy() # (3, 224, 224)
    attr_gray = np.mean(attr, axis=0) # (224, 224)

    # Take absolute value to emphasise magnitude
    attr_gray = np.abs(attr_gray)

    # Contrast stretching
    p1, p99 = np.percentile(attr_gray, (1, 99))
    attr_gray = np.clip((attr_gray - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Smooth the map
    attr_gray = cv2.GaussianBlur(attr_gray, (11, 11), sigmaX=5)

    # Convert to heatmap
    attr_uint8 = np.uint8(attr_gray * 255)
    heatmap = cv2.applyColorMap(attr_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # Overlay
    img_np = np.array(img_pil.resize((224, 224))) / 255.0
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(4,4))
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
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
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
            x, n_segments=50, compactness=10, sigma=1
        )
    )

    # Get both positive and negative contributions
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        hide_rest=False,
        min_weight=0.01 # filter out tiny contributions
    )

    # Convert mask to heatmap
    mask = mask.astype(float)
    mask = cv2.GaussianBlur(mask, (11, 11), sigmaX=5)

    heatmap = cv2.applyColorMap(
        np.uint8((mask - mask.min()) / (mask.max() - mask.min() + 1e-8) * 255),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    overlay = 0.6 * img_np + 0.4 * heatmap
    overlay = np.clip(overlay / 255.0, 0, 1)

    plt.figure(figsize=(4,4))
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
    img, tensor = load_image(img_path)

    variants = {
        "original": img,
        "rotated": img.rotate(20),
        "darker": Image.fromarray((np.array(img) * 0.6).astype(np.uint8)),
        "brighter": Image.fromarray((np.array(img) * 1.4).clip(0,255).astype(np.uint8)),
    }

    for name, variant in variants.items():
        _, t = load_image(img_path)
        save_path = XAI_DIR / "robustness" / f"{Path(img_path).stem}_{name}_gradcam.jpg"
        gradcam(t, variant, save_path)
    
    print(f"[ROBUSTNESS] Completed → {img_path}")

# -----------------------------
# Top error explanations
# -----------------------------
def explain_top_errors():
    with open(TOP_ERRORS_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"[TOP ERRORS] Total top errors: {len(rows)}")

    # Only take first 5 errors
    rows = rows[:5]

    print(f"[TOP ERRORS] Processing {len(rows)} items")

    for i, row in enumerate(rows, start=1):
        fname = row["filename"]
        true_label = row.get("true", "unknown")
        pred_label = row.get("predicted", "unknown")

        print(f"[TOP ERRORS] ({i}/{len(rows)}) {fname} — true: {true_label}, pred: {pred_label}")

        # Find image in original dataset
        img_path = None
        for folder in Path("../dataset").iterdir():
            candidate = folder / fname
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"[TOP ERRORS] WARNING: Could not find {fname} in dataset")
            continue

        # Load image
        img, tensor = load_image(img_path)
        if img is None:
            print(f"[TOP ERRORS] WARNING: Could not load {fname}")
            continue

        # Create folder name with true/pred labels
        folder_name = f"{Path(fname).stem}_true{true_label}_pred{pred_label}"
        error_dir = XAI_DIR / "top_errors" / folder_name
        error_dir.mkdir(parents=True, exist_ok=True)

        # Save all XAI outputs inside this folder
        gradcam(tensor, img, error_dir / "gradcam.jpg")
        integrated_gradients(tensor, img, error_dir / "ig.jpg")
        shap_explain(tensor, img, error_dir / "shap.jpg")
        lime_explain(img, error_dir / "lime.jpg")

    print("[TOP ERRORS] Completed")

# -----------------------------
# Main
# -----------------------------

def clear_xai_folders():
    for folder in [
        "gradcam",
        "integrated_gradients",
        "shap",
        "lime",
        "robustness",
        "top_errors",
        "gradcam_per_fruit"
    ]:
        path = XAI_DIR / folder
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Running XAI ===")
    skip_counter=0
    analysis_counter=0
    clear_xai_folders()

    # Pick a few random test images
    test_images = list(Path(TEST_DIR).rglob("*.*"))
    random.shuffle(test_images)
    test_images = test_images[:5]

    for img_path in test_images:
        print(f"[MAIN] Running XAI on: {img_path}")
        if not str(img_path).lower().endswith((".jpg", ".jpeg", ".png")):
            #print(f"[MAIN] Skipping unsupported file: {img_path}")
            skip_counter+=1
            continue
        img, tensor = load_image(img_path)
        if img is None:
            skip_counter+=1
            continue
        
        analysis_counter+=1
        print("[XAI] Running Grad-CAM...")
        gradcam(tensor, img, XAI_DIR / "gradcam" / f"{Path(img_path).stem}_gradcam.jpg")
        
        print("[XAI] Running Integrated Gradients...")
        integrated_gradients(tensor, img, XAI_DIR / "integrated_gradients" / f"{Path(img_path).stem}_ig.jpg")
        
        print("[XAI] Running SHAP...")
        shap_explain(tensor, img, XAI_DIR / "shap" / f"{Path(img_path).stem}_shap.jpg")
        
        print("[XAI] Running LIME...")
        lime_explain(img, XAI_DIR / "lime" / f"{Path(img_path).stem}_lime.jpg")
        
        print("[XAI] Running Robustness tests...")
        robustness_explain(img_path)

    print("[XAI] Running Grad-CAM one per fruit...")
    gradcam_one_per_fruit()
    
    print("[XAI] Explaining top errors...")
    explain_top_errors()

    print(f"=== XAI complete. Check the xai/ folder ===")
    print(f"=== Images skipped: {skip_counter} with {analysis_counter} analysed ===")

if __name__ == "__main__":
    main()