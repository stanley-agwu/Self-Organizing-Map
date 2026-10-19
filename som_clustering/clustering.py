"""Second level of two-level SOM clustering, plus validity and confidence measures.

Level 1 is the SOM codebook. Level 2 groups the prototypes into K clusters
(Vesanto & Alhoniemi, 2000). A sample's cluster is the cluster of its BMU.
K is chosen on validation data with standard internal validity indices;
activity labels are used only for external evaluation, never for choices.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

CHUNK_SIZE = 8192


class PrototypeClusterer:
    """Clusters the prototypes of units that won at least one training sample.

    Units without hits sit between data clouds and would distort the clusters;
    they get the label of the nearest cluster centroid for display only.
    """

    def __init__(self, weights, hits, method="ward", seed=0):
        self.weights = np.asarray(weights, dtype=float)
        self.active = np.asarray(hits) > 0
        self.method = method
        self.seed = seed
        if method == "ward":
            # One tree serves every K.
            self._tree = linkage(self.weights[self.active], method="ward")
        elif method != "kmeans":
            raise ValueError(f"Unknown clustering method {method!r}")

    def unit_labels(self, k):
        """Cluster label (0..k-1) for every SOM unit."""
        active_weights = self.weights[self.active]
        if self.method == "ward":
            labels = fcluster(self._tree, t=k, criterion="maxclust") - 1
        else:
            labels = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit_predict(active_weights)

        # Renumber so labels are 0..n-1 even if fewer than k clusters came out.
        _, labels = np.unique(labels, return_inverse=True)
        centroids = np.vstack([active_weights[labels == c].mean(axis=0) for c in range(labels.max() + 1)])

        unit_labels = np.empty(len(self.weights), dtype=int)
        unit_labels[self.active] = labels
        if (~self.active).any():
            unit_labels[~self.active] = cdist(self.weights[~self.active], centroids).argmin(axis=1)
        return unit_labels


def internal_indices(X, labels, silhouette_sample, seed):
    """Silhouette (higher is better), Davies-Bouldin (lower), Calinski-Harabasz (higher)."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    sample_size = min(silhouette_sample, len(X)) if silhouette_sample else None
    return {
        "silhouette": float(silhouette_score(X, labels, sample_size=sample_size, random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def select_k(index_table):
    """Choose K by the best mean rank of the three internal indices.

    Ranks avoid the scale problems of adding normalised scores, and each index
    counts once. Ties go to the smaller K.

    Args:
        index_table: DataFrame with columns k, silhouette, davies_bouldin, calinski_harabasz
    Returns:
        (chosen K, table with rank columns added)
    """
    table = index_table.copy()
    table["rank_silhouette"] = table["silhouette"].rank(ascending=False)
    table["rank_davies_bouldin"] = table["davies_bouldin"].rank(ascending=True)
    table["rank_calinski_harabasz"] = table["calinski_harabasz"].rank(ascending=False)
    table["mean_rank"] = table[["rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz"]].mean(axis=1)
    best = table.sort_values(["mean_rank", "k"]).iloc[0]
    return int(best["k"]), table


def external_indices(true_labels, cluster_labels):
    """Agreement between clusters and activity labels (evaluation only)."""
    table = pd.crosstab(cluster_labels, true_labels)
    return {
        "adjusted_rand": float(adjusted_rand_score(true_labels, cluster_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "homogeneity": float(homogeneity_score(true_labels, cluster_labels)),
        "completeness": float(completeness_score(true_labels, cluster_labels)),
        "purity": float(table.max(axis=1).sum() / table.to_numpy().sum()),
    }


def contingency_table(true_labels, cluster_labels):
    """Frames per (cluster, activity)."""
    return pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(true_labels, name="activity"),
    )


def per_sample_silhouette(X, labels, sample_size, seed):
    """Silhouette value per sample on a seeded subsample (returns indices, values)."""
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(X), size=min(sample_size, len(X)), replace=False))
    return indices, silhouette_samples(X[indices], np.asarray(labels)[indices])


# ----------------------------------------------------------------------
# Confidence of an assignment
# ----------------------------------------------------------------------
def fit_confidence_reference(bmu_distances_fit):
    """Reference distances from training data (never from the data being scored)."""
    return {
        "reference_distance": float(np.percentile(bmu_distances_fit, 95)),
        "novelty_distance": float(np.percentile(bmu_distances_fit, 99)),
    }


def bmu_confidence(bmu_distances, reference_distance):
    """1 at a prototype, 0 at or beyond the training 95th-percentile BMU distance."""
    return np.clip(1 - np.asarray(bmu_distances) / reference_distance, 0.0, 1.0)


def cluster_margin(X, weights, unit_labels, bmu_indices):
    """How clearly a sample belongs to its cluster rather than the next one.

    margin = 1 - d_own / d_other, where d_own is the distance to the BMU (which
    is in the sample's own cluster) and d_other the distance to the nearest unit
    of any other cluster. 0 means on the border, values near 1 mean deep inside.
    """
    X = np.asarray(X, dtype=float)
    unit_labels = np.asarray(unit_labels)
    own_cluster = unit_labels[bmu_indices]
    margins = np.empty(len(X))
    for start in range(0, len(X), CHUNK_SIZE):
        stop = start + CHUNK_SIZE
        distances = cdist(X[start:stop], weights)
        own = own_cluster[start:stop]
        d_own = distances[np.arange(len(own)), bmu_indices[start:stop]]
        other = np.where(unit_labels[None, :] == own[:, None], np.inf, distances)
        d_other = other.min(axis=1)
        margins[start:stop] = np.where(np.isfinite(d_other), 1 - d_own / np.maximum(d_other, 1e-12), 1.0)
    return margins
