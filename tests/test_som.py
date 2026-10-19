import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from som_clustering.config import SOMConfig
from som_clustering.som import SelfOrganizingMap, heuristic_map_shape

from tests.helpers import blobs


class GridTests(unittest.TestCase):
    def test_four_neighbourhood(self):
        som = SelfOrganizingMap(3, 4)
        self.assertTrue(som.adjacent[0, 1])
        self.assertTrue(som.adjacent[0, 4])
        self.assertFalse(som.adjacent[0, 5])  # diagonal
        self.assertFalse(som.adjacent[3, 4])  # end of row 0 vs start of row 1
        self.assertEqual(som.adjacent.sum(axis=1).max(), 4)

    def test_sigma_schedule_reaches_end_value(self):
        som = SelfOrganizingMap(6, 10, SOMConfig(n_epochs=10, n_finetune=5, sigma_end=1.0))
        schedule = som.sigma_schedule()
        self.assertAlmostEqual(schedule[0], 5.0)
        self.assertAlmostEqual(schedule[9], 1.0)
        np.testing.assert_allclose(schedule[10:], 1.0)
        self.assertTrue((np.diff(schedule[:10]) < 0).all())

    def test_heuristic_shape_respects_cap(self):
        X, _ = blobs(n_per_blob=3000)
        rows, cols = heuristic_map_shape(X, max_units=100)
        self.assertLessEqual(rows * cols, 110)
        self.assertLessEqual(rows, cols)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = blobs()

    def _train(self, algorithm):
        config = SOMConfig(n_epochs=10, n_finetune=5, algorithm=algorithm)
        return SelfOrganizingMap(5, 6, config, seed=3).fit(self.X, verbose=False)

    def test_batch_training_reduces_error_and_is_deterministic(self):
        first = self._train("batch")
        second = self._train("batch")
        self.assertLess(first.history[-1]["fit_qe"], 0.5 * first.history[0]["fit_qe"])
        np.testing.assert_allclose(first.weights, second.weights)

    def test_online_training_reduces_error(self):
        start = SelfOrganizingMap(5, 6)
        start.initialize_pca(self.X)
        online = self._train("online")
        batch = self._train("batch")
        self.assertLess(online.quantization_error(self.X), 0.85 * start.quantization_error(self.X))
        self.assertLess(online.quantization_error(self.X), 1.1 * batch.quantization_error(self.X))

    def test_bmu_matches_brute_force(self):
        som = self._train("batch")
        indices, distances = som.bmu(self.X, k=2)
        full = cdist(self.X, som.weights)
        np.testing.assert_array_equal(indices[:, 0], full.argmin(axis=1))
        np.testing.assert_allclose(distances[:, 0], full.min(axis=1))
        np.testing.assert_allclose(distances[:, 1], np.sort(full, axis=1)[:, 1])
        self.assertAlmostEqual(som.quantization_error(self.X), full.min(axis=1).mean())
        self.assertEqual(som.hits(self.X).sum(), len(self.X))

    def test_topographic_error_definition(self):
        som = self._train("batch")
        indices = som.bmu(self.X, k=2)[0]
        expected = np.mean(~som.adjacent[indices[:, 0], indices[:, 1]])
        self.assertAlmostEqual(som.topographic_error(self.X), expected)

    def test_save_and_load(self):
        som = self._train("batch")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            som.save(path, unit_labels=np.arange(som.n_units), reference={"a": 1.0})
            loaded, extra = SelfOrganizingMap.load(path)
        np.testing.assert_allclose(loaded.weights, som.weights)
        self.assertEqual((loaded.rows, loaded.cols), (5, 6))
        np.testing.assert_array_equal(extra["unit_labels"], np.arange(30))
        self.assertEqual(extra["reference"], {"a": 1.0})
        self.assertEqual(len(loaded.history), len(som.history))


if __name__ == "__main__":
    unittest.main()
