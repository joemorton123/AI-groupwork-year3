import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from cluster_utils import list_images, build_feature_matrix, kmeans_cluster

DATASET_DIR = "../dataset"


def get_rotten_folders(root):
    return [f for f in os.listdir(root) if f.endswith("__Rotten")]


def map_cluster_to_grade_DE(cluster_id, cluster_means):
    # darker → E, lighter → D
    sorted_clusters = sorted(cluster_means, key=lambda x: x[1])  # ascending
    mapping = {}
    grades = ["E", "D"]  # darkest = E, lighter = D
    for grade, (cid, _) in zip(grades, sorted_clusters):
        mapping[cid] = grade
    return mapping


def main():
    rotten_folders = get_rotten_folders(DATASET_DIR)
    all_paths = []
    folder_for_path = {}

    for folder in rotten_folders:
        folder_path = Path(DATASET_DIR) / folder
        imgs = list_images(folder_path)
        for p in imgs:
            all_paths.append(p)
            folder_for_path[p] = folder

    print(f"Found {len(all_paths)} rotten images.")

    features, valid_paths = build_feature_matrix(all_paths)
    print(f"Using {len(valid_paths)} valid images for clustering.")

    labels = kmeans_cluster(features, n_clusters=2)

    cluster_brightness = defaultdict(list)
    for path, label in zip(valid_paths, labels):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cluster_brightness[label].append(gray.mean())

    cluster_means = [(cid, float(np.mean(vals)))
                     for cid, vals in cluster_brightness.items()]
    cluster_to_grade = map_cluster_to_grade_DE(None, cluster_means)

    print("Cluster → grade mapping:", cluster_to_grade)

    for path, label in zip(valid_paths, labels):
        old_folder = folder_for_path[path]
        fruit = old_folder.split("__")[0]
        grade = cluster_to_grade[label]  # D/E
        new_folder = Path(DATASET_DIR) / f"{fruit}__{grade}"
        new_folder.mkdir(exist_ok=True)
        dest = new_folder / Path(path).name
        os.replace(path, dest)

    print("Rotten images clustered into D/E folders.")


if __name__ == "__main__":
    main()