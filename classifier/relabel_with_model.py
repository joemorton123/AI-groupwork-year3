import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import shutil

DATASET_DIR = "../dataset"
OUTPUT_DIR = "../dataset_refined"
UNCERTAIN_DIR = "../uncertain"
MODEL_PATH = "model_AtoE.pth"

GRADES = ["A", "B", "C", "D", "E"]

def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict(model, img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    conf, cls = torch.max(probs, dim=0)
    return GRADES[cls], float(conf)

def main():
    model = load_model()

    # Clean old outputs
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)
    if Path(UNCERTAIN_DIR).exists():
        shutil.rmtree(UNCERTAIN_DIR)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(UNCERTAIN_DIR).mkdir(exist_ok=True)

    for folder in os.listdir(DATASET_DIR):
        src_folder = Path(DATASET_DIR) / folder
        if not src_folder.is_dir():
            continue

        for img in src_folder.iterdir():
            if not img.is_file():
                continue

            grade, conf = predict(model, img)

            if conf < 0.55:
                shutil.copy(img, Path(UNCERTAIN_DIR) / img.name)
                continue

            # NEW: grade-only folder
            dest_folder = Path(OUTPUT_DIR) / grade
            dest_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dest_folder / img.name)

    print("Re‑prediction complete.")

if __name__ == "__main__":
    main()