"""
Dataset structure analysis utility.

Purpose
-------
Scans a dataset directory organised into subfolders, extracts metadata about files,
produces summary counts, saves tabular and text outputs, and generates bar charts.

Expected folder naming convention
---------------------------------
Each subfolder inside `dataset_dir` is expected to follow one of these patterns:

1. "<fruit>__<grade>"
   Example: "Apple__GradeA"

2. "<fruit>"
   Example: "Banana"

If no grade is present in the folder name, the grade is recorded as "Unknown".

Generated outputs
-----------------
1. CSV file listing all discovered files:
   analysis/results/dataset_file_list.csv

2. Text summary report:
   analysis/results/dataset_analysis.txt

3. Plot images:
   analysis/plots/
"""

import os
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Directory configuration
# -----------------------------
DATASET_DIR = "dataset"
ANALYSIS_DIR = "analysis"
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")

# Ensure output directories exist before analysis begins.
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def analyse_dataset(dataset_dir=DATASET_DIR):
    """
    Analyse the dataset folder structure and file distribution.

    The function walks through each subfolder in the dataset directory, interprets
    folder names as fruit/grade labels, records file metadata, and produces:

    - A DataFrame containing one row per file
    - A CSV export of all file entries
    - A text summary report
    - Bar chart visualisations for key distributions

    Parameters
    ----------
    dataset_dir : str, optional
        Path to the root dataset directory. Defaults to `DATASET_DIR`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing one row per discovered file with the columns:
        - fruit
        - grade
        - folder
        - filename
        - extension
        - path

    Notes
    -----
    Folder names are split using the first occurrence of "__".
    For example:
        "Apple__GradeA" -> fruit="Apple", grade="GradeA"

    If a folder name does not contain "__", the full folder name is used as the
    fruit label and the grade is set to "Unknown".
    """
    # Counters for dataset summary statistics.
    fruit_counts = Counter()
    grade_counts = Counter()
    extension_counts = Counter()
    folder_summary = defaultdict(int)

    # Stores one metadata record per discovered file.
    all_rows = []

    # Iterate through each entry in the dataset root.
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        # Skip non-directory entries at the dataset root level.
        if not os.path.isdir(folder_path):
            continue

        # Extract fruit and grade from folder naming convention.
        if "__" in folder:
            fruit, grade = folder.split("__", 1)
        else:
            fruit, grade = folder, "Unknown"

        # Iterate through files inside the current folder.
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # Skip nested directories or non-file entries.
            if not os.path.isfile(file_path):
                continue

            # Extract file extension without the leading dot.
            _, ext = os.path.splitext(file)
            ext = ext.lower().replace(".", "")

            # Update summary counts.
            fruit_counts[fruit] += 1
            grade_counts[grade] += 1
            extension_counts[ext] += 1
            folder_summary[folder] += 1

            # Store file-level metadata for later export.
            all_rows.append(
                {
                    "fruit": fruit,
                    "grade": grade,
                    "folder": folder,
                    "filename": file,
                    "extension": ext,
                    "path": file_path,
                }
            )

    # Create a DataFrame from all collected file metadata.
    df = pd.DataFrame(all_rows)

    # -------------------------------------------------
    # Save CSV containing all discovered dataset entries
    # -------------------------------------------------
    df.to_csv(os.path.join(RESULTS_DIR, "dataset_file_list.csv"), index=False)

    # ------------------------
    # Save text summary report
    # ------------------------
    summary_path = os.path.join(RESULTS_DIR, "dataset_analysis.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== DATASET SUMMARY ==========\n\n")
        f.write(f"Total images: {len(df)}\n\n")

        f.write("Images per fruit:\n")
        for fruit, count in fruit_counts.items():
            f.write(f"  {fruit}: {count}\n")

        f.write("\nImages per grade:\n")
        for grade, count in grade_counts.items():
            f.write(f"  {grade}: {count}\n")

        f.write("\nImages per folder:\n")
        for folder, count in folder_summary.items():
            f.write(f"  {folder}: {count}\n")

        f.write("\nFile extensions found:\n")
        for ext, count in extension_counts.items():
            f.write(f"  .{ext}: {count}\n")

        f.write("\n=====================================\n")

    # Apply Seaborn styling to all plots.
    sns.set(style="whitegrid")

    def save_bar_chart(counter_data, title, xlabel, filename):
        """
        Save a bar chart from a dictionary-like count structure.

        Parameters
        ----------
        counter_data : dict or collections.Counter
            Mapping of category labels to counts.
        title : str
            Chart title.
        xlabel : str
            X-axis label.
        filename : str
            Output filename for the saved plot.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counter_data.keys()), y=list(counter_data.values()))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.close()

    # ----------------
    # Generate plots
    # ----------------
    save_bar_chart(fruit_counts, "Images per Fruit", "Fruit", "fruit_distribution.png")
    save_bar_chart(grade_counts, "Images per Grade", "Grade", "grade_distribution.png")
    save_bar_chart(
        extension_counts,
        "Images per File Extension",
        "Extension",
        "extension_distribution.png",
    )
    save_bar_chart(folder_summary, "Images per Folder", "Folder", "folder_distribution.png")

    print(f"\nAnalysis complete! Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")

    return df


if __name__ == "__main__":
    # Run the analysis when the script is executed directly.
    df = analyse_dataset()

    # Display a small preview of the collected file metadata.
    print("\nPreview of dataset entries:")
    print(df.head())