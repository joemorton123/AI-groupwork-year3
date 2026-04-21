"""
Comprehensive evaluation pipeline for a three-class fruit-grade classifier.

Purpose
-------
This script evaluates a trained EfficientNet-B0 classifier on the test split of the
dataset and produces a set of quantitative and qualitative analysis outputs.

Generated outputs
-----------------
The following files are written to the `results/` directory:

- metrics.txt
    Overall accuracy, classification report, and confusion matrix

- confidence_histogram.png
    Confidence distribution for correct versus incorrect predictions

- calibration_curve.png
    Reliability diagram comparing predicted confidence with observed accuracy

- all_errors.csv
    All misclassified samples, sorted by confidence

- top_errors.csv
    Highest-confidence misclassifications

- error_summary.txt
    Summary of total and high-confidence errors

- per_fruit_accuracy.csv
    Accuracy grouped by fruit type using the original dataset structure

- robustness_test.txt
    Predictions for one sample under several simple image perturbations

- model_comparison.txt
    Accuracy comparison between the first-pass and final models

- random_samples.txt
    Predictions for a random sample of test images

Evaluation scope
----------------
The script performs:

1. Standard test-set evaluation
2. Confidence analysis
3. Calibration analysis
4. Misclassification analysis
5. Per-fruit performance breakdown
6. Simple robustness checks
7. Comparison of first-pass and final models
8. Random-sample qualitative inspection

Assumptions
-----------
- The test dataset is arranged for `torchvision.datasets.ImageFolder`.
- Both saved models use EfficientNet-B0 with a three-class output head.
- Test labels correspond to the grade order: A, B, C.
- The original dataset still exists and can be used to recover fruit type from file names.

Side effects
------------
- Creates a `results/` directory if it does not already exist.
- Writes multiple text, CSV, and PNG files into that directory.
"""

import csv
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------
# Paths
# -----------------------------
TEST_DIR = "../dataset_split/test"
MODEL_PATH = "../classifier/model_AtoC.pth"
FIRST_PASS_MODEL_PATH = "../classifier/model_first_pass.pth"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GRADES = ["A", "B", "C"]


def load_model(path):
    """
    Load a trained EfficientNet-B0 classifier from disk.

    Parameters
    ----------
    path : str or Path
        Path to the saved model state dictionary.

    Returns
    -------
    torch.nn.Module or None
        Loaded model in evaluation mode if successful; otherwise `None`.

    Notes
    -----
    - The classifier head is replaced with a three-class linear layer before loading.
    - Weights are loaded onto CPU using `map_location="cpu"`.
    - `model.eval()` is applied so that inference behaviour is deterministic.
    """
    print(f"\n[INFO] Loading model from: {path}")
    if not Path(path).exists():
        print(f"[ERROR] Model file not found: {path}")
        return None

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)

    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print("[SUCCESS] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def save_text(filename, content):
    """
    Save a text string into the results directory.

    Parameters
    ----------
    filename : str
        Name of the output text file.
    content : str
        Text content to be written.

    Returns
    -------
    None
    """
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SAVED] {filename}")


def sample_random_images(model, n=10):
    """
    Run inference on a random sample of test images and save the predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded classification model in evaluation mode.
    n : int, optional
        Number of random test images to sample. Default is 10.

    Returns
    -------
    None

    Notes
    -----
    - Images are sampled from all files found under `TEST_DIR`.
    - The output contains filename, predicted grade, and confidence score.
    """
    print("\n[STEP] Running random sample predictions...")
    output = []
    all_images = list(Path(TEST_DIR).rglob("*.*"))

    if len(all_images) == 0:
        print("[WARNING] No images found for random sampling.")
        return

    samples = random.sample(all_images, min(n, len(all_images)))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    for img_path in samples:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, cls = torch.max(probs, dim=0)

        line = f"{img_path.name}: predicted {GRADES[cls.item()]} ({conf:.2f})"
        output.append(line)

    save_text("random_samples.txt", "\n".join(output))


def plot_confidence_histogram(confidences, correct_mask):
    """
    Plot confidence distributions for correct and incorrect predictions.

    Parameters
    ----------
    confidences : numpy.ndarray
        Confidence score for each prediction.
    correct_mask : numpy.ndarray
        Boolean mask indicating whether each prediction is correct.

    Returns
    -------
    None

    Notes
    -----
    Two histograms are overlaid:
    - correct predictions
    - incorrect predictions
    """
    print("[STEP] Plotting confidence histogram...")
    plt.figure(figsize=(8, 5))
    plt.hist(
        [confidences[correct_mask], confidences[~correct_mask]],
        bins=20,
        label=["Correct", "Incorrect"],
        color=["green", "red"],
        alpha=0.7,
    )
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confidence_histogram.png")
    plt.close()
    print("[SAVED] confidence_histogram.png")


def plot_calibration_curve(confidences, correct_mask, n_bins=10):
    """
    Plot a calibration curve (reliability diagram).

    Parameters
    ----------
    confidences : array-like
        Predicted confidence scores.
    correct_mask : array-like
        Boolean values indicating prediction correctness.
    n_bins : int, optional
        Number of confidence bins. Default is 10.

    Returns
    -------
    None

    Method
    ------
    Confidence scores are divided into equal-width bins. For each bin, observed
    accuracy is computed and plotted against the bin centre.

    Notes
    -----
    Perfect calibration lies on the diagonal y = x.
    """
    print("[STEP] Plotting calibration curve...")

    confidences = np.array(confidences)
    correct_mask = np.array(correct_mask)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    counts = []

    for i in range(n_bins):
        bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        bin_correct = correct_mask[bin_mask]

        if len(bin_correct) > 0:
            accuracies.append(np.mean(bin_correct))
            counts.append(len(bin_correct))
        else:
            accuracies.append(np.nan)
            counts.append(0)

    plt.figure(figsize=(7, 6))
    plt.plot(bin_centers, accuracies, marker="o", label="Model calibration")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "calibration_curve.png")
    plt.close()

    print("[SAVED] calibration_curve.png")


def save_top_errors(test_paths, preds, labels, confidences, n=20):
    """
    Save complete and top-ranked misclassification records.

    Parameters
    ----------
    test_paths : list[pathlib.Path]
        File paths for test samples, aligned with predictions.
    preds : list[int]
        Predicted class indices.
    labels : list[int]
        True class indices.
    confidences : list[float]
        Confidence scores for the predicted classes.
    n : int, optional
        Number of highest-confidence errors to save separately. Default is 20.

    Returns
    -------
    None

    Outputs
    -------
    - all_errors.csv
    - top_errors.csv
    - error_summary.txt
    """
    print("[STEP] Saving top misclassifications...")

    errors = [
        (p, pr, lb, cf)
        for p, pr, lb, cf in zip(test_paths, preds, labels, confidences)
        if pr != lb
    ]

    total_errors = len(errors)
    print(f"[INFO] Total misclassifications: {total_errors}")

    errors_100 = [e for e in errors if e[3] == 1.0]
    errors_99 = [e for e in errors if e[3] >= 0.99]

    print(f"[INFO] Misclassifications with 100% confidence: {len(errors_100)}")
    print(f"[INFO] Misclassifications with >=99% confidence: {len(errors_99)}")

    errors_sorted = sorted(errors, key=lambda x: x[3], reverse=True)

    with open(RESULTS_DIR / "all_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted", "true", "confidence"])
        for p, pr, lb, cf in errors_sorted:
            writer.writerow([p.name, GRADES[pr], GRADES[lb], f"{cf:.2f}"])
    print("[SAVED] all_errors.csv")

    with open(RESULTS_DIR / "top_errors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted", "true", "confidence"])
        for p, pr, lb, cf in errors_sorted[:n]:
            writer.writerow([p.name, GRADES[pr], GRADES[lb], f"{cf:.2f}"])
    print(f"[SAVED] top_errors.csv (top {n} errors)")

    summary = (
        f"Total misclassifications: {total_errors}\n"
        f"Misclassifications with 100% confidence: {len(errors_100)}\n"
        f"Misclassifications with >=99% confidence: {len(errors_99)}\n"
        f"Top errors saved: {n}\n"
    )
    save_text("error_summary.txt", summary)


def save_per_fruit_accuracy(test_paths, preds, labels):
    """
    Calculate and save accuracy grouped by fruit type.

    Fruit type is recovered by indexing the original dataset folder structure. The
    original folder format is expected to follow:

        <fruit>__<grade>

    Parameters
    ----------
    test_paths : list[pathlib.Path]
        File paths for test samples.
    preds : list[int]
        Predicted class indices.
    labels : list[int]
        True class indices.

    Returns
    -------
    None

    Output
    ------
    per_fruit_accuracy.csv
    """
    print("[STEP] Calculating TRUE per-fruit accuracy using original dataset...")

    ORIGINAL_DATASET = Path("../dataset")

    print("[INFO] Indexing original dataset...")
    filename_to_fruit = {}

    for folder in ORIGINAL_DATASET.iterdir():
        if folder.is_dir() and "__" in folder.name:
            fruit = folder.name.split("__")[0]
            for img_path in folder.iterdir():
                filename_to_fruit[img_path.name] = fruit

    print(f"[INFO] Indexed {len(filename_to_fruit)} original images.")

    fruit_stats = {}

    for p, pr, lb in zip(test_paths, preds, labels):
        fname = p.name

        if fname not in filename_to_fruit:
            print(f"[WARNING] Could not find original fruit type for: {fname}")
            continue

        fruit = filename_to_fruit[fname]

        if fruit not in fruit_stats:
            fruit_stats[fruit] = {"correct": 0, "total": 0}

        fruit_stats[fruit]["total"] += 1
        if pr == lb:
            fruit_stats[fruit]["correct"] += 1

    out_path = RESULTS_DIR / "per_fruit_accuracy.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fruit", "accuracy", "correct", "total"])

        for fruit, stats in sorted(fruit_stats.items()):
            acc = stats["correct"] / stats["total"]
            writer.writerow([fruit, f"{acc:.3f}", stats["correct"], stats["total"]])

    print("[SAVED] per_fruit_accuracy.csv")


def robustness_test(model, img_path):
    """
    Run simple robustness checks on a single image.

    The selected image is evaluated in four conditions:
    - original
    - rotated by 20 degrees
    - darkened
    - brightened

    Parameters
    ----------
    model : torch.nn.Module
        Loaded classification model.
    img_path : pathlib.Path
        Path to the test image to be perturbed.

    Returns
    -------
    None

    Output
    ------
    robustness_test.txt
    """
    print(f"[STEP] Running robustness test on: {img_path.name}")
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def predict(img):
        """
        Predict grade and confidence for a PIL image object.

        Parameters
        ----------
        img : PIL.Image.Image
            Input image.

        Returns
        -------
        tuple[str, float]
            Predicted grade label and confidence score.
        """
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, cls = torch.max(probs, dim=0)
        return GRADES[cls.item()], float(conf)

    results = []

    pred, conf = predict(img)
    results.append(f"Original: {pred} ({conf:.2f})")

    pred_r, conf_r = predict(img.rotate(20))
    results.append(f"Rotated 20 degrees: {pred_r} ({conf_r:.2f})")

    enhancer = ImageEnhance.Brightness(img)
    pred_d, conf_d = predict(enhancer.enhance(0.6))
    results.append(f"Darker: {pred_d} ({conf_d:.2f})")

    pred_b, conf_b = predict(enhancer.enhance(1.4))
    results.append(f"Brighter: {pred_b} ({conf_b:.2f})")

    save_text("robustness_test.txt", "\n".join(results))


def compare_models(first_model, final_model, test_loader):
    """
    Compare first-pass and final model accuracy on the same test set.

    Parameters
    ----------
    first_model : torch.nn.Module
        First-pass model.
    final_model : torch.nn.Module
        Final refined model.
    test_loader : torch.utils.data.DataLoader
        Test-set loader.

    Returns
    -------
    None

    Output
    ------
    model_comparison.txt
    """
    print("[STEP] Comparing first-pass vs final model...")

    def evaluate(model):
        """
        Compute simple classification accuracy for a model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate.

        Returns
        -------
        float
            Mean accuracy over the test set.
        """
        preds, labels = [], []
        with torch.no_grad():
            for imgs, lbs in test_loader:
                logits = model(imgs)
                pr = torch.argmax(logits, dim=1)
                preds.extend(pr.numpy())
                labels.extend(lbs.numpy())
        return np.mean(np.array(preds) == np.array(labels))

    acc_first = evaluate(first_model)
    acc_final = evaluate(final_model)

    content = (
        f"First-pass accuracy: {acc_first:.4f}\n"
        f"Final accuracy:      {acc_final:.4f}\n"
    )
    save_text("model_comparison.txt", content)


def main():
    """
    Run the full evaluation pipeline.

    Workflow
    --------
    1. Load the test dataset.
    2. Load the final trained model.
    3. Generate predictions and confidence scores.
    4. Save standard classification metrics.
    5. Run extended analysis:
       - confidence histogram
       - calibration curve
       - error analysis
       - per-fruit accuracy
    6. Run a robustness test on one random image.
    7. Compare first-pass and final models if both are available.
    8. Save predictions for a random sample of test images.

    Returns
    -------
    None
    """
    print("\n=== Starting Evaluation ===")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("[INFO] Loading test dataset...")
    test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"[INFO] Test samples: {len(test_ds)}")

    model = load_model(MODEL_PATH)
    if model is None:
        print("[FATAL] Cannot continue without final model.")
        return

    print("[STEP] Running predictions...")
    all_preds = []
    all_labels = []
    all_confidences = []
    test_paths = [Path(p[0]) for p in test_ds.samples]

    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            conf, cls = torch.max(probs, dim=1)

            all_preds.extend(cls.numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(conf.numpy())

    print("[STEP] Saving metrics...")
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n\n"
        f"{classification_report(all_labels, all_preds, target_names=GRADES)}\n"
        f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}\n"
    )
    save_text("metrics.txt", metrics_text)

    print("[STEP] Running extended analysis...")
    correct_mask = np.array(all_preds) == np.array(all_labels)
    plot_confidence_histogram(np.array(all_confidences), correct_mask)
    plot_calibration_curve(all_confidences, correct_mask)
    save_top_errors(test_paths, all_preds, all_labels, all_confidences)
    save_per_fruit_accuracy(test_paths, all_preds, all_labels)

    print("[STEP] Running robustness test...")
    robustness_test(model, random.choice(test_paths))

    print("[STEP] Checking for first-pass model...")
    if Path(FIRST_PASS_MODEL_PATH).exists():
        first_model = load_model(FIRST_PASS_MODEL_PATH)
        if first_model is not None:
            compare_models(first_model, model, test_loader)
        else:
            save_text("model_comparison.txt", "First-pass model found but could not be loaded.")
    else:
        print("[INFO] First-pass model not found. Skipping comparison.")
        save_text("model_comparison.txt", "First-pass model not found. Comparison skipped.")

    print("[STEP] Running random sampling...")
    sample_random_images(model, n=10)

    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: {RESULTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()