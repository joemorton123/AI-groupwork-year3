"""
Cluster healthy fruit images into quality grades.

Purpose
-------
This script scans dataset folders whose names end with "__Healthy", extracts image
features, groups the healthy images into three clusters using K-Means, estimates the
average brightness of each cluster, and maps those clusters to quality grades:

- brightest cluster  -> Grade A
- middle cluster     -> Grade B
- darkest cluster    -> Grade C

Each image is then moved from its original healthy folder into a new folder named
using the pattern:

    <fruit>__<grade>

Example:
    Apple__Healthy  ->  Apple__A, Apple__B, Apple__C

Assumptions
-----------
- The dataset root contains folders named like "<fruit>__Healthy".
- `cluster_utils.py` provides:
    - list_images(folder_path)
    - build_feature_matrix(image_paths)
    - kmeans_cluster(features, n_clusters)
- Brightness is used as a proxy for grade ordering.
- Only images successfully processed by `build_feature_matrix` are clustered and moved.

Side effects
------------
- Creates new grade folders if they do not already exist.
- Moves image files on disk using `os.replace`.
"""

import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from cluster_utils import list_images, build_feature_matrix, kmeans_cluster


# Root dataset directory containing fruit folders.
DATASET_DIR = "../dataset"


def get_healthy_folders(root):
    """
    Return all folder names in the dataset root that represent healthy fruit images.

    Parameters
    ----------
    root : str or Path
        Path to the dataset root directory.

    Returns
    -------
    list[str]
        Folder names ending with "__Healthy".

    Notes
    -----
    Only the folder name is returned, not the full path.
    """
    return [f for f in os.listdir(root) if f.endswith("__Healthy")]


def map_cluster_to_grade(cluster_means):
    """
    Map cluster IDs to letter grades based on average cluster brightness.

    Clusters are sorted by brightness in descending order:
    - brightest cluster becomes grade "A"
    - next brightest becomes grade "B"
    - darkest becomes grade "C"

    Parameters
    ----------
    cluster_means : list[tuple[int, float]]
        A list of `(cluster_id, mean_brightness)` pairs.

    Returns
    -------
    dict[int, str]
        Mapping from cluster ID to grade label.

    Example
    -------
    Input:
        [(2, 180.4), (0, 145.1), (1, 120.7)]

    Output:
        {2: "A", 0: "B", 1: "C"}
    """
    sorted_clusters = sorted(cluster_means, key=lambda x: x[1], reverse=True)
    mapping = {}
    grades = ["A", "B", "C"]

    for grade, (cid, _) in zip(grades, sorted_clusters):
        mapping[cid] = grade

    return mapping


def main():
    """
    Run the healthy-image clustering and grading pipeline.

    Workflow
    --------
    1. Find all folders ending with "__Healthy".
    2. Collect all image paths from those folders.
    3. Build a feature matrix for valid images.
    4. Cluster the images into three groups using K-Means.
    5. Compute average brightness for each cluster.
    6. Map clusters to grades A/B/C based on brightness.
    7. Move each valid image into a new folder named "<fruit>__<grade>".

    Returns
    -------
    None

    Notes
    -----
    - Only paths returned in `valid_paths` are used after feature extraction.
    - Images that fail feature extraction are excluded from clustering and movement.
    - `os.replace` will overwrite the destination if a file with the same name exists.
    """
    healthy_folders = get_healthy_folders(DATASET_DIR)

    # Stores every discovered healthy image path.
    all_paths = []

    # Maps each image path to its original folder name, so the fruit type can be recovered later.
    folder_for_path = {}

    # Collect all healthy image paths across all matching folders.
    for folder in healthy_folders:
        folder_path = Path(DATASET_DIR) / folder
        imgs = list_images(folder_path)

        for p in imgs:
            all_paths.append(p)
            folder_for_path[p] = folder

    print(f"Found {len(all_paths)} healthy images.")

    # Build features only for images that can be successfully processed.
    features, valid_paths = build_feature_matrix(all_paths)
    print(f"Using {len(valid_paths)} valid images for clustering.")

    # Cluster images into three groups.
    labels = kmeans_cluster(features, n_clusters=3)

    # Compute mean grayscale brightness for each cluster.
    # This is later used to rank clusters from brightest to darkest.
    cluster_brightness = defaultdict(list)

    for path, label in zip(valid_paths, labels):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cluster_brightness[label].append(gray.mean())

    cluster_means = [
        (cid, float(np.mean(vals)))
        for cid, vals in cluster_brightness.items()
    ]

    cluster_to_grade = map_cluster_to_grade(cluster_means)
    print("Cluster → grade mapping:", cluster_to_grade)

    # Move each clustered image into its new grade folder.
    for path, label in zip(valid_paths, labels):
        old_folder = folder_for_path[path]
        fruit = old_folder.split("__")[0]
        grade = cluster_to_grade[label]

        new_folder = Path(DATASET_DIR) / f"{fruit}__{grade}"
        new_folder.mkdir(exist_ok=True)

        dest = new_folder / Path(path).name
        os.replace(path, dest)

    print("Healthy images clustered into A/B/C folders.")


if __name__ == "__main__":
    main()