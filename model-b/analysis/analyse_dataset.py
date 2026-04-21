import os
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR = "dataset"
ANALYSIS_DIR = "analysis"
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def analyse_dataset(dataset_dir=DATASET_DIR):
    fruit_counts = Counter()
    grade_counts = Counter()
    extension_counts = Counter()
    folder_summary = defaultdict(int)

    all_rows = []

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        if "__" in folder:
            fruit, grade = folder.split("__", 1)
        else:
            fruit, grade = folder, "Unknown"

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if not os.path.isfile(file_path):
                continue

            # Extract extension
            _, ext = os.path.splitext(file)
            ext = ext.lower().replace(".", "")

            fruit_counts[fruit] += 1
            grade_counts[grade] += 1
            extension_counts[ext] += 1
            folder_summary[folder] += 1

            all_rows.append({
                "fruit": fruit,
                "grade": grade,
                "folder": folder,
                "filename": file,
                "extension": ext,
                "path": file_path
            })

    df = pd.DataFrame(all_rows)

    # Save CSV of all image entries
    df.to_csv(os.path.join(RESULTS_DIR, "dataset_file_list.csv"), index=False)

    # Save text summary
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

    # Generate plots
    sns.set(style="whitegrid")

    def save_plot(data, title, xlabel, filename):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(data.keys()), y=list(data.values()))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.close()
    
    def save_bar_chart(counter_data, title, xlabel, filename):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counter_data.keys()), y=list(counter_data.values()))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.close()

    # Save results
    save_plot(fruit_counts, "Images per Fruit", "Fruit", "fruit_distribution.png")
    save_plot(grade_counts, "Images per Grade", "Grade", "grade_distribution.png")
    save_plot(extension_counts, "Images per File Extension", "Extension", "extension_distribution.png")
    save_plot(folder_summary, "Images per Folder", "Folder", "folder_distribution.png")

    # Save image results
    save_bar_chart(fruit_counts, "Images per Fruit", "Fruit", "fruit_distribution.png")
    save_bar_chart(grade_counts, "Images per Grade", "Grade", "grade_distribution.png")
    save_bar_chart(extension_counts, "Images per File Extension", "Extension", "extension_distribution.png")
    save_bar_chart(folder_summary, "Images per Folder", "Folder", "folder_distribution.png")

    print(f"\nAnalysis complete! Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")

    return df


if __name__ == "__main__":
    df = analyse_dataset()
    print("\nPreview of dataset entries:")
    print(df.head())