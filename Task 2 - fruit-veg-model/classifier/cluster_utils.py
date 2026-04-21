"""
Utility functions for image discovery, feature extraction, and K-Means clustering.

Purpose
-------
This module provides helper functions used for image-based clustering tasks in the
fruit-grading workflow. It supports:

- locating valid image files in a folder
- extracting a fixed-length feature vector from each image
- building a feature matrix for multiple images
- clustering the resulting features with K-Means

Feature representation
----------------------
Each image is resized to a fixed resolution and represented by a normalised
3D colour histogram over the B, G, and R channels. This produces a compact
summary of colour distribution, which can be used for unsupervised grouping.

Dependencies
------------
- OpenCV (`cv2`) for image loading, resizing, and histogram extraction
- NumPy for matrix construction
- scikit-learn for K-Means clustering
"""

import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


def list_images(folder):
    """
    Return all valid image file paths in the given folder.

    Parameters
    ----------
    folder : str or Path
        Path to the folder containing image files.

    Returns
    -------
    list[str]
        A list of image file paths as strings.

    Notes
    -----
    Only files with supported image extensions are included. Subdirectories are ignored.

    Supported extensions
    --------------------
    - .jpg
    - .jpeg
    - .png
    - .bmp
    - .tif
    - .tiff
    - .jfif
    - .webp
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif", ".webp"}
    paths = []

    for f in os.listdir(folder):
        p = Path(folder) / f
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))

    return paths


def extract_feature(image_path, size=(128, 128)):
    """
    Extract a normalised colour histogram feature vector from an image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    size : tuple[int, int], optional
        Target size used when resizing the image before feature extraction.
        Default is (128, 128).

    Returns
    -------
    numpy.ndarray or None
        A flattened normalised histogram feature vector if the image is read
        successfully, otherwise `None`.

    Method
    ------
    The image is:
    1. loaded from disk using OpenCV
    2. resized to a fixed size
    3. converted into a 3-channel colour histogram with 8 bins per channel
    4. normalised and flattened into a one-dimensional feature vector

    Notes
    -----
    - OpenCV loads images in BGR channel order.
    - The histogram shape is 8 × 8 × 8, producing 512 features after flattening.
    - Returning `None` allows unreadable or corrupted images to be skipped safely.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, size)

    # Compute a 3D colour histogram across the B, G, and R channels.
    hist = cv2.calcHist(
        [img],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    )

    # Normalise the histogram and flatten it into a 1D feature vector.
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def build_feature_matrix(image_paths):
    """
    Build a feature matrix from a list of image paths.

    Parameters
    ----------
    image_paths : list[str]
        List of image file paths.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        A tuple containing:
        - a NumPy array of extracted feature vectors
        - a list of image paths that were successfully processed

    Notes
    -----
    Images that cannot be read or processed are excluded from the feature matrix.
    The returned `valid_paths` list preserves alignment with the feature rows.
    This means:
        feature_matrix[i] corresponds to valid_paths[i]
    """
    features = []
    valid_paths = []

    for p in image_paths:
        feat = extract_feature(p)
        if feat is not None:
            features.append(feat)
            valid_paths.append(p)

    return np.array(features), valid_paths


def kmeans_cluster(features, n_clusters):
    """
    Cluster image feature vectors using K-Means.

    Parameters
    ----------
    features : numpy.ndarray
        A 2D feature matrix where each row represents one image.
    n_clusters : int
        The number of clusters to form.

    Returns
    -------
    numpy.ndarray
        Cluster labels for each feature vector.

    Notes
    -----
    - A fixed random seed (`random_state=42`) is used for reproducibility.
    - `n_init="auto"` allows scikit-learn to choose an appropriate number of
      centroid initialisations.
    - The returned labels align with the row order of the input `features`.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(features)
    return labels