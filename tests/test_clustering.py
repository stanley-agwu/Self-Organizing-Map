import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from som_clustering.clustering import (
    PrototypeClusterer,
    bmu_confidence,
    cluster_margin,
    external_indices,
    internal_indices,
    select_k,
)
from som_clustering.config import SOMConfig
from som_clustering.som import SelfOrganizingMap
from som_clustering.temporal import run_lengths, stability_summary, stable_mask

from tests.helpers import blobs


class TwoLevelClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = blobs()
        cls.som = SelfOrganizingMap(6, 6, SOMConfig(n_epochs=10, n_finetune=5), seed=0).fit(cls.X, verbose=False)
        cls.bmu = cls.som.bmu(cls.X)[0][:, 0]
        cls.hits = np.bincount(cls.bmu, minlength=cls.som.n_units)

    def test_recovers_known_clusters(self):
        for method in ("ward", "kmeans"):
            clusterer = PrototypeClusterer(self.som.weights, self.hits, method=method, seed=0)
            labels = clusterer.unit_labels(3)[self.bmu]
            self.assertGreater(adjusted_rand_score(self.y, labels), 0.95, method)

    def test_select_k_finds_three(self):
        clusterer = PrototypeClusterer(self.som.weights, self.hits, seed=0)
        rows = [{"k": k, **internal_indices(self.X, clusterer.unit_labels(k)[self.bmu], 0, 0)} for k in range(2, 7)]
        chosen, table = select_k(pd.DataFrame(rows))
        self.assertEqual(chosen, 3)
        self.assertIn("mean_rank", table)

    def test_units_without_hits_still_get_labels(self):
        hits = self.hits.copy()
        hits[0] = 0
        labels = PrototypeClusterer(self.som.weights, hits, seed=0).unit_labels(3)
        self.assertEqual(len(labels), self.som.n_units)
        self.assertTrue(set(labels) <= {0, 1, 2})

    def test_cluster_margin_range(self):
        unit_labels = PrototypeClusterer(self.som.weights, self.hits, seed=0).unit_labels(3)
        margin = cluster_margin(self.X, self.som.weights, unit_labels, self.bmu)
        self.assertTrue(((margin >= 0) & (margin <= 1)).all())
        self.assertGreater(margin.mean(), 0.5)  # well separated blobs


class ScoreTests(unittest.TestCase):
    def test_external_indices_perfect_match(self):
        truth = np.array([1, 1, 2, 2, 3, 3])
        scores = external_indices(truth, np.array([5, 5, 0, 0, 7, 7]))
        self.assertAlmostEqual(scores["adjusted_rand"], 1.0)
        self.assertAlmostEqual(scores["purity"], 1.0)

    def test_bmu_confidence_clipped(self):
        np.testing.assert_allclose(bmu_confidence([0.0, 1.0, 3.0], 2.0), [1.0, 0.5, 0.0])


class TemporalTests(unittest.TestCase):
    def test_run_lengths(self):
        np.testing.assert_array_equal(run_lengths([1, 1, 2, 2, 2, 1]), [2, 2, 3, 3, 3, 1])

    def test_runs_do_not_cross_tracks(self):
        labels = np.array([0] * 6 + [0] * 6)
        tracks = np.array(["a"] * 6 + ["b"] * 6)
        self.assertFalse(stable_mask(labels, tracks, 10).any())
        self.assertTrue(stable_mask(labels, np.array(["a"] * 12), 10).all())

    def test_stability_summary(self):
        labels = np.array([0] * 30 + [1] * 30)
        summary = stability_summary(labels, np.array(["a"] * 60), 10)
        self.assertEqual(summary["stable_fraction"], 1.0)
        self.assertEqual(summary["mean_run_length_frames"], 30.0)
        self.assertAlmostEqual(summary["switches_per_second"], 0.5)


if __name__ == "__main__":
    unittest.main()
