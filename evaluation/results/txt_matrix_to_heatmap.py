import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_confusion_matrix(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    matrix_lines = []
    capture = False

    for line in lines:
        if line.strip().startswith("[["):
            capture = True

        if capture:
            matrix_lines.append(line.strip())
            if line.strip().endswith("]]"):
                break

    cleaned = []
    for row in matrix_lines:
        nums = re.findall(r"\d+", row)
        cleaned.append([int(n) for n in nums])

    return np.array(cleaned)

def plot_confusion_matrix_with_percentages(cm, labels, out_path):
    # Normalise by true class (row-wise)
    row_sums = cm.sum(axis=1, keepdims=True)
    percentages = cm / row_sums

    # Create annotation text "count (xx.x%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({percentages[i, j]*100:.1f}%)"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        percentages,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor="grey"
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix — Raw Counts and Normalised Percentages")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SAVED] Confusion matrix heatmap → {out_path}")

def main():
    txt_path = Path("metrics.txt")
    out_path = Path("confusion_matrix.png")

    if not txt_path.exists():
        print("metrics.txt not found.")
        return

    cm = load_confusion_matrix(txt_path)
    print("Loaded confusion matrix:")
    print(cm)

    labels = ["A", "B", "C"]
    plot_confusion_matrix_with_percentages(cm, labels, out_path)

if __name__ == "__main__":
    main()