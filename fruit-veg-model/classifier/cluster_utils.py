import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif", ".webp"}
    paths = []
    for f in os.listdir(folder):
        p = Path(folder) / f
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    return paths


def extract_feature(image_path, size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, size)
    # simple colour histogram feature
    hist = cv2.calcHist([img], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def build_feature_matrix(image_paths):
    features = []
    valid_paths = []
    for p in image_paths:
        feat = extract_feature(p)
        if feat is not None:
            features.append(feat)
            valid_paths.append(p)
    return np.array(features), valid_paths


def kmeans_cluster(features, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(features)
    return labels