"""Self-Organizing Map (Kohonen) with a rectangular grid and Euclidean distance.

Training follows the usual two phases:
* ordering: the neighbourhood width sigma shrinks exponentially from
  max(rows, cols) / 2 to ``sigma_end`` over ``n_epochs``;
* convergence: up to ``n_finetune`` more epochs at ``sigma_end``, stopped early
  once the relative weight change ||W_t - W_{t-1}|| / ||W_{t-1}|| < ``tol``.

Two update rules are available:
* "batch" (default): Kohonen's batch map, w_i = sum_x h_{c(x),i} x / sum_x h_{c(x),i};
* "online": w_i += alpha(t) * h_{c,i}(t) * (x - w_i), with alpha decaying
  exponentially from ``alpha_start`` to ``alpha_end`` over all steps.

Both minimise Euclidean distortion, so the best-matching unit (BMU) is always
found with Euclidean distance.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .config import SOMConfig

CHUNK_SIZE = 8192


def heuristic_map_shape(X, max_units):
    """Map size from the Vesanto heuristic (about 5 * sqrt(N) units).

    The side ratio follows the square root of the ratio of the two largest
    principal-component variances. The total is capped at ``max_units``.
    """
    n_units = min(int(np.ceil(5 * np.sqrt(len(X)))), max_units)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    ratio = np.sqrt(eigenvalues[0] / max(eigenvalues[1], 1e-12))
    rows = max(2, int(round(np.sqrt(n_units / ratio))))
    cols = max(2, int(round(n_units / rows)))
    return min(rows, cols), max(rows, cols)


class SelfOrganizingMap:
    def __init__(self, rows, cols, config=None, seed=0):
        self.rows = int(rows)
        self.cols = int(cols)
        self.config = config or SOMConfig()
        self.seed = seed
        self.weights = None
        self.history = []

        r, c = np.divmod(np.arange(self.n_units), self.cols)
        self.grid = np.column_stack([r, c]).astype(float)
        self.grid_sq_dist = cdist(self.grid, self.grid, "sqeuclidean")
        # 4-neighbourhood on the rectangular grid
        self.adjacent = self.grid_sq_dist == 1

    @property
    def n_units(self):
        return self.rows * self.cols

    # ------------------------------------------------------------------
    # Initialisation and schedules
    # ------------------------------------------------------------------
    def initialize_pca(self, X):
        """Linear initialisation on the plane of the first two principal components."""
        mean = X.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(X, rowvar=False))
        order = np.argsort(eigenvalues)[::-1]
        std = np.sqrt(np.maximum(eigenvalues[order[:2]], 0))
        pc1, pc2 = eigenvectors[:, order[0]], eigenvectors[:, order[1]]

        # The longer grid side follows the first component, spanning about +-2 std.
        long_axis, short_axis = (1, 0) if self.cols >= self.rows else (0, 1)
        sides = (self.rows, self.cols)
        a = self._spread(self.grid[:, long_axis], sides[long_axis])
        b = self._spread(self.grid[:, short_axis], sides[short_axis])
        self.weights = mean + 2 * (a[:, None] * std[0] * pc1 + b[:, None] * std[1] * pc2)

    @staticmethod
    def _spread(index, size):
        return np.zeros_like(index) if size == 1 else 2 * index / (size - 1) - 1

    def sigma_schedule(self):
        """Neighbourhood width for each epoch (ordering phase, then convergence)."""
        cfg = self.config
        sigma_start = max(max(self.rows, self.cols) / 2, cfg.sigma_end)
        steps = np.arange(cfg.n_epochs) / max(cfg.n_epochs - 1, 1)
        ordering = sigma_start * (cfg.sigma_end / sigma_start) ** steps
        return np.r_[ordering, np.full(cfg.n_finetune, cfg.sigma_end)]

    def neighbourhood(self, sigma):
        return np.exp(-self.grid_sq_dist / (2 * sigma**2))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, X_val=None, verbose=True):
        X = np.asarray(X, dtype=float)
        X_val = None if X_val is None else np.asarray(X_val, dtype=float)
        rng = np.random.default_rng(self.seed)
        if self.weights is None:
            self.initialize_pca(X)

        sigmas = self.sigma_schedule()
        total_steps = self.config.n_epochs * len(X)
        step = 0
        self.history = []

        for epoch, sigma in enumerate(sigmas):
            previous = self.weights.copy()
            if self.config.algorithm == "batch":
                self._batch_epoch(X, sigma)
            elif self.config.algorithm == "online":
                step = self._online_epoch(X, sigma, rng, step, total_steps)
            else:
                raise ValueError(f"Unknown algorithm {self.config.algorithm!r}")

            change = np.linalg.norm(self.weights - previous) / max(np.linalg.norm(previous), 1e-12)
            record = {"epoch": epoch + 1, "sigma": float(sigma), "relative_change": float(change)}
            record.update(self._metrics(X, "fit"))
            if X_val is not None:
                record.update(self._metrics(X_val, "val"))
            self.history.append(record)

            if verbose and (epoch % 10 == 0 or epoch == len(sigmas) - 1):
                val = f", val QE {record['val_qe']:.4f}" if X_val is not None else ""
                print(f"  epoch {epoch + 1:3d}  sigma {sigma:.2f}  QE {record['fit_qe']:.4f}"
                      f"  TE {record['fit_te']:.3f}{val}  dW {change:.2e}")

            in_convergence_phase = epoch >= self.config.n_epochs
            if in_convergence_phase and change < self.config.tol:
                if verbose:
                    print(f"  converged at epoch {epoch + 1} (dW {change:.2e} < tol {self.config.tol})")
                break
        return self

    def _batch_epoch(self, X, sigma):
        bmu = self.bmu(X)[0][:, 0]
        sums = np.zeros_like(self.weights)
        np.add.at(sums, bmu, X)
        counts = np.bincount(bmu, minlength=self.n_units).astype(float)

        h = self.neighbourhood(sigma)
        numerator = h @ sums
        denominator = h @ counts
        updated = denominator > 1e-12
        self.weights[updated] = numerator[updated] / denominator[updated, None]

    def _online_epoch(self, X, sigma, rng, step, total_steps):
        cfg = self.config
        h_all = self.neighbourhood(sigma)
        for x in X[rng.permutation(len(X))]:
            progress = min(step / max(total_steps - 1, 1), 1.0)
            alpha = cfg.alpha_start * (cfg.alpha_end / cfg.alpha_start) ** progress
            winner = np.argmin(((self.weights - x) ** 2).sum(axis=1))
            self.weights += alpha * h_all[winner][:, None] * (x - self.weights)
            step += 1
        return step

    def _metrics(self, X, prefix):
        indices, distances = self.bmu(X, k=2)
        return {
            f"{prefix}_qe": float(distances[:, 0].mean()),
            f"{prefix}_te": float(self._topographic_error(indices)),
        }

    # ------------------------------------------------------------------
    # Mapping and quality measures
    # ------------------------------------------------------------------
    def bmu(self, X, k=1):
        """Indices and Euclidean distances of the k closest units, nearest first."""
        X = np.asarray(X, dtype=float)
        indices = np.empty((len(X), k), dtype=int)
        distances = np.empty((len(X), k))
        for start in range(0, len(X), CHUNK_SIZE):
            chunk = X[start:start + CHUNK_SIZE]
            sq = cdist(chunk, self.weights, "sqeuclidean")
            nearest = np.argpartition(sq, k - 1, axis=1)[:, :k] if k < self.n_units else np.argsort(sq, axis=1)
            nearest_sq = np.take_along_axis(sq, nearest, axis=1)
            order = np.argsort(nearest_sq, axis=1)
            indices[start:start + len(chunk)] = np.take_along_axis(nearest, order, axis=1)[:, :k]
            distances[start:start + len(chunk)] = np.sqrt(np.take_along_axis(nearest_sq, order, axis=1))[:, :k]
        return indices, distances

    def quantization_error(self, X):
        """Mean Euclidean distance from each sample to its BMU."""
        return float(self.bmu(X)[1][:, 0].mean())

    def topographic_error(self, X):
        """Share of samples whose first and second BMUs are not grid neighbours."""
        return float(self._topographic_error(self.bmu(X, k=2)[0]))

    def _topographic_error(self, indices):
        return np.mean(~self.adjacent[indices[:, 0], indices[:, 1]])

    def hits(self, X):
        return np.bincount(self.bmu(X)[0][:, 0], minlength=self.n_units)

    def u_matrix(self):
        """Mean distance from each unit's weights to its 4 grid neighbours, shaped (rows, cols)."""
        distances = cdist(self.weights, self.weights)
        neighbour_count = self.adjacent.sum(axis=1)
        mean_distance = (distances * self.adjacent).sum(axis=1) / np.maximum(neighbour_count, 1)
        return mean_distance.reshape(self.rows, self.cols)

    # ------------------------------------------------------------------
    # Persistence (small .npz, no pickle)
    # ------------------------------------------------------------------
    def save(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "rows": self.rows,
            "cols": self.cols,
            "seed": self.seed,
            "config": asdict(self.config),
            "history": self.history,
        }
        arrays = {key: np.asarray(value) for key, value in extra.items() if not isinstance(value, dict)}
        meta["extra"] = {key: value for key, value in extra.items() if isinstance(value, dict)}
        np.savez_compressed(path, weights=self.weights, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path):
        """Return (som, extra) where extra holds the arrays and dicts saved with it."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            som = cls(meta["rows"], meta["cols"], SOMConfig(**meta["config"]), meta["seed"])
            som.weights = data["weights"]
            som.history = meta["history"]
            extra = {key: data[key] for key in data.files if key not in ("weights", "meta")}
        extra.update(meta.get("extra", {}))
        return som, extra
