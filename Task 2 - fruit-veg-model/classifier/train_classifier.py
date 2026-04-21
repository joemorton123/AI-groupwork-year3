"""
Train an EfficientNet-B0 image classifier for fruit grade prediction.

Purpose
-------
This script trains a three-class image classifier using a dataset already split into
training and validation folders. The model uses EfficientNet-B0 with ImageNet
pretrained weights and replaces the final classification layer so that it predicts
three grade labels:

    A, B, C

The trained model weights are saved to disk for later inference or second-pass
refinement.

Expected dataset structure
--------------------------
Training data:
    ../dataset_split/train/A
    ../dataset_split/train/B
    ../dataset_split/train/C

Validation data:
    ../dataset_split/val/A
    ../dataset_split/val/B
    ../dataset_split/val/C

Pass configuration
------------------
To switch between first-pass and second-pass training, change only `SAVE_PATH`:

First pass:
    SAVE_PATH = "model_first_pass.pth"

Second pass:
    SAVE_PATH = "model_AtoC.pth"

Assumptions
-----------
- The dataset has already been split into train and validation folders.
- Folder names correspond exactly to the class labels expected by `ImageFolder`.
- EfficientNet-B0 is suitable for the image classification task.
- Three output classes are required.

Side effects
------------
- Loads image data from disk.
- Trains a neural network model.
- Saves model weights to the configured output path.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Directory containing the training split.
DATASET_DIR = "../dataset_split/train"

# Directory containing the validation split.
VAL_DIR = "../dataset_split/val"

# Output file for the first training pass.
SAVE_PATH = "model_first_pass.pth"

# Output file for the second training pass.
# SAVE_PATH = "model_AtoC.pth"


# Training hyperparameters.
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 3  # Grade labels: A, B, C


def get_loaders():
    """
    Create DataLoaders for the training and validation datasets.

    Images are resized to 224 × 224 and converted to tensors before being passed
    into the model. The training dataset is shuffled to improve stochastic training
    behaviour, while the validation dataset is loaded in deterministic order.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list[str]]
        A tuple containing:
        - training DataLoader
        - validation DataLoader
        - class names discovered by `ImageFolder`

    Notes
    -----
    - `ImageFolder` automatically assigns class indices based on subfolder names.
    - The class order is determined alphabetically by folder name.
    - No normalisation is currently applied beyond tensor conversion.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(DATASET_DIR, transform=transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader, train_ds.classes


def main():
    """
    Run the model training pipeline.

    Workflow
    --------
    1. Load the training and validation datasets.
    2. Initialise an EfficientNet-B0 model with ImageNet pretrained weights.
    3. Replace the final classifier layer with a new linear layer for three classes.
    4. Move the model to GPU if available, otherwise use CPU.
    5. Train the model for the configured number of epochs.
    6. Save the trained model weights to disk.

    Returns
    -------
    None

    Notes
    -----
    - Cross-entropy loss is used for multi-class classification.
    - Adam is used as the optimiser.
    - The validation loader is created but not currently used for evaluation.
    - Only model weights are saved, not the full model object.
    """
    train_loader, val_loader, classes = get_loaders()

    # Load EfficientNet-B0 with ImageNet pretrained weights.
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Replace the final classification layer so that the model predicts A/B/C.
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    # Use GPU if available; otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Define the loss function and optimiser.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop.
    for epoch in range(EPOCHS):
        model.train()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} complete.")

    # Save only the learned model parameters.
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()