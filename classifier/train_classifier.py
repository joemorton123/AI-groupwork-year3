import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATASET_DIR = "../dataset_split/train"
VAL_DIR = "../dataset_split/val"
SAVE_PATH = "model_AtoC.pth"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 3  # A/B/C

def get_loaders():
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
    train_loader, val_loader, classes = get_loaders()

    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()