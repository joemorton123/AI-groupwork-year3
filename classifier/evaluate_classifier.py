import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

TEST_DIR = "../dataset_split/test"
MODEL_PATH = "model_AtoE.pth"
GRADES = ["A", "B", "C", "D", "E"]

def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = load_model()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print("\n=== Overall Accuracy ===")
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"{accuracy:.4f}")

    print("\n=== Per‑Grade Metrics ===")
    print(classification_report(all_labels, all_preds, target_names=GRADES))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()