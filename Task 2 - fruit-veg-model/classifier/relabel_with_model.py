"""
Refine fruit-grade labels using a trained EfficientNet classifier.

Purpose
-------
This script re-predicts grade labels for images stored in the dataset directory
using a trained EfficientNet-B0 model. Based on prediction confidence, each image
is copied into one of two destinations:

1. Refined dataset folder
   - Images with sufficiently high confidence are copied into:
       ../dataset_refined/A
       ../dataset_refined/B
       ../dataset_refined/C

2. Uncertain folder
   - Images with low confidence are copied into:
       ../uncertain

This supports a semi-automated labelling workflow in which confident predictions
are accepted automatically and uncertain predictions can later be reviewed manually.

Assumptions
-----------
- The trained model checkpoint exists at `model_AtoC.pth`.
- The model predicts exactly three classes corresponding to grades A, B, and C.
- The dataset root contains folders of images, and all files inside those folders
  are intended for prediction.
- EfficientNet-B0 architecture matches the saved model weights.

Side effects
------------
- Deletes any existing contents of `../dataset_refined` and `../uncertain`.
- Recreates those directories from scratch.
- Copies images into new output folders using `shutil.copy`.
"""

import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Root directory containing source image folders.
DATASET_DIR = "../dataset"

# Directory for confidently re-labelled images.
OUTPUT_DIR = "../dataset_refined"

# Directory for low-confidence predictions requiring manual review.
UNCERTAIN_DIR = "../uncertain"

# Saved PyTorch model checkpoint.
MODEL_PATH = "model_AtoC.pth"

# Class labels corresponding to model output indices.
GRADES = ["A", "B", "C"]


def load_model():
    """
    Load the trained EfficientNet-B0 model for grade prediction.

    The model architecture is recreated first, then the final classifier layer is
    adjusted to output three classes corresponding to grades A, B, and C. The saved
    weights are then loaded from disk, and the model is placed into evaluation mode.

    Returns
    -------
    torch.nn.Module
        The loaded EfficientNet-B0 model ready for inference.

    Notes
    -----
    - `weights=None` is used because the model weights are loaded from a custom checkpoint.
    - `map_location="cpu"` ensures the model can be loaded on systems without a GPU.
    - `model.eval()` disables training-specific behaviour such as dropout.
    """
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(model, img_path):
    """
    Predict the grade and confidence score for a single image.

    Parameters
    ----------
    model : torch.nn.Module
        The trained classification model in evaluation mode.
    img_path : str or Path
        Path to the image file to be classified.

    Returns
    -------
    tuple[str, float]
        A tuple containing:
        - predicted grade label ("A", "B", or "C")
        - confidence score as a float in the range [0, 1]

    Method
    ------
    1. Resize the image to 224 × 224.
    2. Convert the image to a tensor.
    3. Add a batch dimension.
    4. Run a forward pass without gradient tracking.
    5. Convert logits to probabilities using softmax.
    6. Select the highest-probability class and confidence score.

    Notes
    -----
    - The image is converted to RGB before preprocessing.
    - `cls.item()` is used to convert the predicted class tensor into a Python integer.
    """
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
    return GRADES[cls.item()], float(conf)


def main():
    """
    Run the dataset re-prediction and refinement pipeline.

    Workflow
    --------
    1. Load the trained classification model.
    2. Remove any existing refined and uncertain output directories.
    3. Recreate clean output directories.
    4. Iterate through all image files in the dataset folders.
    5. Predict the grade and confidence for each image.
    6. Copy low-confidence images into the uncertain folder.
    7. Copy high-confidence images into grade-specific refined folders.

    Confidence rule
    ---------------
    - confidence < 0.55  -> copy image to `../uncertain`
    - confidence >= 0.55 -> copy image to `../dataset_refined/<grade>`

    Returns
    -------
    None

    Notes
    -----
    - Files are copied, not moved, so the original dataset remains unchanged.
    - Existing output folders are deleted at the start of execution.
    - The confidence threshold can be adjusted depending on how strict the
      automatic relabelling process should be.
    """
    model = load_model()

    # Remove previous outputs so the refinement process starts from a clean state.
    if Path(OUTPUT_DIR).exists():
        shutil.rmtree(OUTPUT_DIR)
    if Path(UNCERTAIN_DIR).exists():
        shutil.rmtree(UNCERTAIN_DIR)

    # Recreate output directories.
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(UNCERTAIN_DIR).mkdir(exist_ok=True)

    # Process each folder in the source dataset.
    for folder in os.listdir(DATASET_DIR):
        src_folder = Path(DATASET_DIR) / folder
        if not src_folder.is_dir():
            continue

        # Process each file in the current source folder.
        for img in src_folder.iterdir():
            if not img.is_file():
                continue

            grade, conf = predict(model, img)

            # Send low-confidence predictions to manual review.
            if conf < 0.55:
                shutil.copy(img, Path(UNCERTAIN_DIR) / img.name)
                continue

            # Save confident predictions into the refined dataset.
            dest_folder = Path(OUTPUT_DIR) / grade
            dest_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dest_folder / img.name)

    print("Re-prediction complete.")


if __name__ == "__main__":
    main()