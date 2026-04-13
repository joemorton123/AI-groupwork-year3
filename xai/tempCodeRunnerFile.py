import torch
import torch.nn.functional as F
from torchvision import models, transforms
from captum.attr import IntegratedGradients
from lime import lime_image
import shap
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "../classifier/model_AtoC.pth"
TEST_DIR = "../dataset_split/test"
TOP_ERRORS_CSV = "../evaluation/results/top_errors.csv"

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
def get_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv

def gradcam(img_tensor, img_pil, save_path):
    print(f"[GradCAM] Processing {save_path.name}")

    model.zero_grad()

    # --- AUTO-DETECT LAST CONV LAYER ---
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

    act = activations[0].detach().squeeze(0)      # shape (C, H, W)
    grad = gradients[0].detach().squeeze(0)       # shape (C, H, W)

    # --- FIX: ensure shapes are correct ---
    if act.ndim != 3:
        print(f"[GradCAM] SKIPPED — invalid activation shape: {act.shape}")
        return

    if grad.ndim != 3:
        print(f"[GradCAM] SKIPPED — invalid gradient shape: {grad.shape}")
        return

    # Global average pooling
    weights = grad.mean(dim=(1, 2))    # shape (C)

    # Weighted sum
    cam = (weights[:, None, None] * act).sum(dim=0)  # shape (H, W)

    cam = cam.cpu().numpy()
    cam = np.maximum(cam, 0)

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Resize to 224x224
    cam_resized = cv2.resize(cam, (224, 224))

    # Convert to uint8 single-channel
    cam_uint8 = np.uint8(cam_resized * 255)

    print("CAM DEBUG:", cam_uint8.dtype, cam_uint8.shape)

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Overlay
    img_np = np.array(img_pil.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    cv2.imwrite(str(save_path), overlay[:, :, ::-1])
    print(f"[GradCAM] Saved → {save_path}")

# -----------------------------
# Integrated Gradients
# -----------------------------
def integrated_gradients(img_tensor, img_pil, save_path):
    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(img_tensor)

    logits = model(img_tensor)
    pred_class = logits.argmax().item()

    attributions = ig.attribute(img_tensor, baseline, target=pred_class)

# -----------------------------
# SHAP
# -----------------------------
def shap_explain(img_tensor, img_pil, save_path):
    background = torch.zeros_like(img_tensor)
    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(img_tensor)

    shap.image_plot(shap_values, np.array(img_pil.resize((224, 224))), show=False)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

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
        np.array(img_pil),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        hide_rest=False
    )

    plt.imshow(temp)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

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

# -----------------------------
# Top error explanations
# -----------------------------
def explain_top_errors():
    import csv

    with open(TOP_ERRORS_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        fname = row["filename"]

        # Find image in original dataset
        for folder in Path("../dataset").iterdir():
            candidate = folder / fname
            if candidate.exists():
                img_path = candidate
                break

        img, tensor = load_image(img_path)

        gradcam(tensor, img, XAI_DIR / "top_errors" / f"{fname}_gradcam.jpg")
        integrated_gradients(tensor, img, XAI_DIR / "top_errors" / f"{fname}_ig.jpg")
        shap_explain(tensor, img, XAI_DIR / "top_errors" / f"{fname}_shap.jpg")
        lime_explain(img, XAI_DIR / "top_errors" / f"{fname}_lime.jpg")

# -----------------------------
# Main
# -----------------------------
def main():
    print("=== Running XAI ===")
    skip_counter=0
    analysis_counter=0

    # Pick a few random test images
    test_images = list(Path(TEST_DIR).rglob("*.*"))[:5]

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

    print("[XAI] Explaining top errors...")
    explain_top_errors()

    print(f"=== XAI complete. Check the xai/ folder ===")
    print(f"=== Images skipped: {skip_counter} with {analysis_counter} analyzed ===")

if __name__ == "__main__":
    main()