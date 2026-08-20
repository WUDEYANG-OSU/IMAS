from __future__ import annotations
import numpy as np


def _l2norm_rows(values):
    values = np.asarray(values, np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def spherical_kmeans_fit(z_train: np.ndarray, n_clusters: int, iterations: int = 25,
                         max_train: int = 60_000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed); z = _l2norm_rows(z_train)
    fit = z[rng.choice(len(z), min(len(z), max_train), replace=False)] if len(z) > max_train else z
    k = min(max(2, int(n_clusters)), len(fit)); centers = fit[rng.choice(len(fit), k, replace=False)].copy()
    for _ in range(iterations):
        labels = (fit @ centers.T).argmax(1)
        for cluster in range(k):
            mask = labels == cluster
            centers[cluster] = fit[mask].mean(0) if mask.any() else fit[rng.integers(len(fit))]
        centers = _l2norm_rows(centers)
    return centers


def spherical_kmeans_assign(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return (_l2norm_rows(values) @ _l2norm_rows(centers).T).argmax(1).astype(np.int64)


def build_group_ids_from_clusters(train_clusters: np.ndarray, all_clusters: np.ndarray, min_train_size: int = 100):
    counts = np.bincount(train_clusters, minlength=int(train_clusters.max()) + 1)
    mapping = {int(cluster): i + 1 for i, cluster in enumerate(np.where(counts >= min_train_size)[0])}
    groups = np.asarray([mapping.get(int(cluster), 0) for cluster in all_clusters], np.int64)
    return groups, mapping, counts
