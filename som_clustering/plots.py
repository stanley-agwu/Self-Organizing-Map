"""All figures. Every figure is saved once and closed.

Design rules:
* one measure per axis (no dual y-axes);
* at most three categorical colours on one plot. Clusters (up to 20) are shown
  with numbered regions, small multiples or step lines, never with 20 hues;
* magnitudes use a single blue ramp, light to dark.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from .config import KEPT_JOINTS, SKELETON_EDGES  # noqa: E402

# Reference palette (validated: 3 slots pass CVD/normal-vision checks on light).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BACKGROUND_POINTS = "#d6d5ce"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

JOINT_POSITION = {joint: position for position, joint in enumerate(KEPT_JOINTS)}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _grid_panels(n_panels, n_cols, panel_size=(2.6, 2.6)):
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
                             squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes.flat


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def plot_training_history(history, tol, n_ordering_epochs, path):
    """QE and TE on fit and validation data, sigma, and relative weight change."""
    epochs = [h["epoch"] for h in history]
    has_val = "val_qe" in history[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    panels = [
        (axes[0, 0], "qe", "Quantization error (mean BMU distance)"),
        (axes[0, 1], "te", "Topographic error (share of samples)"),
    ]
    for ax, key, title in panels:
        ax.plot(epochs, [h[f"fit_{key}"] for h in history], color=SERIES[0], label="fit (training)")
        if has_val:
            ax.plot(epochs, [h[f"val_{key}"] for h in history], color=SERIES[1], label="validation")
        ax.set_title(title, loc="left")
        ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [h["sigma"] for h in history], color=SERIES[0])
    ax.set_title("Neighbourhood width σ", loc="left")

    ax = axes[1, 1]
    ax.semilogy(epochs, [h["relative_change"] for h in history], color=SERIES[0])
    ax.axhline(tol, color=MUTED, linestyle="--", linewidth=1)
    ax.text(epochs[0], tol, f" stop threshold {tol:g}", color=INK_SECONDARY, fontsize=8, va="bottom")
    ax.set_title("Relative weight change ‖ΔW‖ / ‖W‖", loc="left")

    for ax in axes.flat:
        ax.axvline(n_ordering_epochs + 0.5, color=AXIS, linewidth=0.8)
        ax.set_xlabel("Epoch")
    axes[1, 0].text(n_ordering_epochs + 0.5, axes[1, 0].get_ylim()[1], " convergence phase →",
                    color=INK_SECONDARY, fontsize=8, va="top")
    fig.suptitle("SOM training history", x=0.06, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_map_size_sweep(table, chosen_units, path):
    """Validation QE and TE against the number of map units (sensitivity only)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in [(axes[0], "val_qe", "Validation QE"), (axes[1], "val_te", "Validation TE")]:
        ax.plot(table["units"], table[key], color=SERIES[0], marker="o", markersize=5)
        ax.axvline(chosen_units, color=AXIS, linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Map units (rows × cols)")
    axes[0].text(chosen_units, axes[0].get_ylim()[1], " used map", color=INK_SECONDARY, fontsize=8, va="top")
    fig.suptitle("Map-size sensitivity (QE always falls as the map grows; not used for selection)",
                 x=0.06, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# Map views
# ----------------------------------------------------------------------
def _cluster_boundaries(ax, labels_grid):
    rows, cols = labels_grid.shape
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and labels_grid[r, c] != labels_grid[r, c + 1]:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=INK, linewidth=1.4)
            if r + 1 < rows and labels_grid[r, c] != labels_grid[r + 1, c]:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=INK, linewidth=1.4)


def _label_clusters(ax, labels_grid):
    for cluster in np.unique(labels_grid):
        r, c = np.argwhere(labels_grid == cluster).mean(axis=0)
        ax.text(c, r, str(cluster), ha="center", va="center", fontsize=9, fontweight="bold", color=INK,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85})


def _map_axes(ax, rows, cols):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)


def plot_u_matrix(u_matrix, unit_labels, path):
    """U-matrix (distance to neighbours) with the second-level cluster regions drawn on it."""
    rows, cols = u_matrix.shape
    labels_grid = np.asarray(unit_labels).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(0.42 * cols + 2.2, 0.42 * rows + 0.8))
    image = ax.imshow(u_matrix, cmap=BLUES, interpolation="nearest")
    _cluster_boundaries(ax, labels_grid)
    _label_clusters(ax, labels_grid)
    _map_axes(ax, rows, cols)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Mean distance to grid neighbours (dark = cluster border)")
    ax.set_title(f"U-matrix and posture clusters ({labels_grid.max() + 1} clusters, {rows}×{cols} map)", loc="left")
    _save(fig, path)


def plot_hit_maps(hit_maps, unit_labels, rows, cols, path):
    """Share of frames won by each unit, one panel per data set, on one shared scale."""
    labels_grid = np.asarray(unit_labels).reshape(rows, cols)
    shares = {name: hits.reshape(rows, cols) / hits.sum() for name, hits in hit_maps.items()}
    vmax = max(share.max() for share in shares.values())
    cell = min(0.2, 3.6 / max(rows, cols))
    fig, axes = plt.subplots(1, len(shares), figsize=(len(shares) * (cell * cols + 0.3) + 1.2, cell * rows + 1.0),
                             squeeze=False, layout="constrained")
    for ax, (name, share) in zip(axes.flat, shares.items()):
        image = ax.imshow(share * 100, cmap=BLUES, vmin=0, vmax=vmax * 100, interpolation="nearest")
        _cluster_boundaries(ax, labels_grid)
        _map_axes(ax, rows, cols)
        ax.set_title(name, loc="left")
    colorbar = fig.colorbar(image, ax=list(axes.flat), fraction=0.03, pad=0.02)
    colorbar.set_label("% of frames whose BMU is this unit")
    fig.suptitle("Hit maps", x=0.02, ha="left", color=INK)
    _save(fig, path)


# ----------------------------------------------------------------------
# Choosing K
# ----------------------------------------------------------------------
def plot_k_selection(table, chosen_k, path):
    """The three internal indices and their mean rank against K, on validation data."""
    panels = [
        ("silhouette", "Silhouette (higher is better)"),
        ("davies_bouldin", "Davies–Bouldin (lower is better)"),
        ("calinski_harabasz", "Calinski–Harabasz (higher is better)"),
        ("mean_rank", "Mean rank of the three (lower is better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, (key, title) in zip(axes.flat, panels):
        ax.plot(table["k"], table[key], color=SERIES[0], marker="o", markersize=4)
        match = table.loc[table["k"] == chosen_k, key]
        if len(match):  # a fixed K may lie outside the scored range
            ax.plot([chosen_k], [match.iloc[0]], marker="o", markersize=9, color=SERIES[1], zorder=3)
            ax.annotate(f"K = {chosen_k}", (chosen_k, match.iloc[0]), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=INK)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Number of clusters K")
        ax.set_xticks(table["k"])
    fig.suptitle("Choosing the number of posture clusters on validation data", x=0.06, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# Clusters in data space
# ----------------------------------------------------------------------
def plot_clusters_pca(X, labels, fit_reference, path, title, max_points=6000, seed=0):
    """Small multiples: each panel highlights one cluster over all data in PCA space.

    PCA is fitted on ``fit_reference`` (training data) so every data set is
    projected onto the same axes.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if len(X) > max_points:
        index = rng.choice(len(X), size=max_points, replace=False)
        X, labels = X[index], labels[index]
    pca = PCA(n_components=2, random_state=seed).fit(fit_reference)
    projected = pca.transform(X)
    clusters = np.unique(labels)

    fig, axes = _grid_panels(len(clusters), n_cols=min(5, len(clusters)))
    for ax, cluster in zip(axes, clusters):
        member = labels == cluster
        ax.scatter(projected[~member, 0], projected[~member, 1], s=2, color=BACKGROUND_POINTS, linewidths=0)
        ax.scatter(projected[member, 0], projected[member, 1], s=3, color=SERIES[0], linewidths=0)
        ax.set_title(f"Cluster {cluster}  ({member.mean():.0%})", loc="left", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    explained = pca.explained_variance_ratio_
    fig.suptitle(f"{title}  —  PCA of training data (PC1 {explained[0]:.0%}, PC2 {explained[1]:.0%} of variance)",
                 x=0.02, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_contingency(table, path, title):
    """Share of each activity's frames that falls in each cluster (columns sum to 100%)."""
    shares = table / table.sum(axis=0)
    fig, ax = plt.subplots(figsize=(0.7 * shares.shape[1] + 2.4, 0.42 * shares.shape[0] + 1.4))
    image = ax.imshow(shares.to_numpy() * 100, cmap=BLUES, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(shares.shape[1]), [f"A{a}" for a in shares.columns])
    ax.set_yticks(range(shares.shape[0]), [str(c) for c in shares.index])
    ax.set_xlabel("Activity (from file name)")
    ax.set_ylabel("Cluster")
    ax.grid(False)
    for (r, c), value in np.ndenumerate(shares.to_numpy()):
        if value >= 0.05:
            ax.text(c, r, f"{value:.0%}", ha="center", va="center", fontsize=7,
                    color=SURFACE if value > 0.55 else INK)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("% of the activity's frames")
    ax.set_title(title, loc="left")
    _save(fig, path)


def _draw_skeleton(ax, joints, horizontal_axis, color):
    for parent, child in SKELETON_EDGES:
        p, c = JOINT_POSITION[parent], JOINT_POSITION[child]
        ax.plot([joints[p, horizontal_axis], joints[c, horizontal_axis]], [joints[p, 1], joints[c, 1]],
                color=color, linewidth=2, solid_capstyle="round")
    ax.scatter(joints[:, horizontal_axis], joints[:, 1], s=10, color=color, zorder=3)


def plot_cluster_postures(cluster_joints, cluster_shares, path):
    """Mean posture of each cluster, front view (x-y) above side view (z-y).

    Coordinates are body-centred (pelvis at 0, hips along x) in torso lengths.
    """
    clusters = sorted(cluster_joints)
    n_cols = min(6, len(clusters))
    n_blocks = int(np.ceil(len(clusters) / n_cols))
    fig, axes = plt.subplots(2 * n_blocks, n_cols, figsize=(2.1 * n_cols, 4.4 * n_blocks), squeeze=False)
    all_joints = np.vstack([cluster_joints[c] for c in clusters])
    limit = np.abs(all_joints).max() * 1.05

    for ax in axes.flat:
        ax.set_visible(False)
    for i, cluster in enumerate(clusters):
        block, col = divmod(i, n_cols)
        for view, horizontal_axis in enumerate([0, 2]):
            ax = axes[2 * block + view, col]
            ax.set_visible(True)
            _draw_skeleton(ax, cluster_joints[cluster], horizontal_axis, SERIES[0])
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if view == 0:
                ax.set_title(f"Cluster {cluster}  ({cluster_shares[cluster]:.0%})", loc="left", fontsize=9)
            ax.set_xlabel("front (x–y)" if view == 0 else "side (z–y)", fontsize=7, color=MUTED)
    fig.suptitle("Mean posture per cluster (training frames)", x=0.02, ha="left", color=INK)
    fig.tight_layout()
    _save(fig, path)


# ----------------------------------------------------------------------
# Test-time behaviour
# ----------------------------------------------------------------------
def plot_track_timelines(track_data, min_run_length, n_clusters, path):
    """Per track: cluster label over time (step line) and assignment confidence below it.

    track_data: list of dicts with keys name, labels, confidence, margin, stable.
    """
    n = len(track_data)
    fig, axes = plt.subplots(2 * n, 1, figsize=(11, 2.6 * n), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1] * n})
    for i, track in enumerate(track_data):
        frames = np.arange(len(track["labels"]))
        ax = axes[2 * i, 0]
        ax.step(frames, track["labels"], where="post", color=SERIES[0], linewidth=1.4)
        unstable = ~track["stable"]
        ax.scatter(frames[unstable], track["labels"][unstable], s=8, color=SERIES[1], zorder=3,
                   label=f"in a run < {min_run_length} frames")
        ax.set_ylabel("Cluster")
        ax.set_title(track["name"], loc="left", fontsize=9)
        ax.set_yticks(range(n_clusters))
        ax.set_ylim(-0.4, n_clusters - 0.6)

        ax = axes[2 * i + 1, 0]
        ax.plot(frames, track["confidence"], color=SERIES[0], label="BMU confidence")
        ax.plot(frames, track["margin"], color=SERIES[1], label="cluster margin")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Score")
        ax.set_xlabel("Frame in track")

    handles = axes[0, 0].get_legend_handles_labels()[0] + axes[1, 0].get_legend_handles_labels()[0]
    labels = axes[0, 0].get_legend_handles_labels()[1] + axes[1, 0].get_legend_handles_labels()[1]
    fig.suptitle("Cluster assignment over time for example test tracks", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.995))
    _save(fig, path)


def plot_score_distributions(scores, path):
    """Distribution of BMU confidence and cluster margin per data set (step histograms)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 1, 41)
    for ax, key, title in [(axes[0], "confidence", "BMU confidence"), (axes[1], "margin", "Cluster margin")]:
        for color, (name, values) in zip(SERIES, scores.items()):
            ax.hist(values[key], bins=bins, histtype="step", density=True, color=color, linewidth=1.6, label=name)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Score (0 = weak, 1 = strong)")
        ax.set_ylabel("Density")
        ax.legend()
    fig.tight_layout()
    _save(fig, path)
