"""Regression checks for parameter preservation and analytical metric definitions."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from QHDRT import QHDRT
from analyze_results import curve_metrics
from plot_selected_heatmaps import display_titles


class SensitivityTests(unittest.TestCase):
    def test_count_metric_and_partial_auc(self):
        result = curve_metrics([0, 1, 2, 2, 2, 2, 2, 2, 2, 2], 2)
        self.assertEqual(result['failures_30'], 2)
        self.assertEqual(result['recall_30'], 1)
        self.assertAlmostEqual(result['normalized_auc_30'], 1 / 3)
        self.assertAlmostEqual(result['normalized_auc'], .8)
        self.assertTrue(np.isnan(curve_metrics([0] * 10, 0)['normalized_auc_30']))
        self.assertTrue(np.isnan(curve_metrics([1, 1], 1)['normalized_auc_30']))

    def test_partial_auc_matches_paper_trapezoidal_formula(self):
        histories = [[0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                     [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]]
        computed, expected = [], []
        for counts in histories:
            b, f = len(counts) * 30 // 100, counts[-1]
            c = [0] + counts
            expected.append(sum((c[k-1] + c[k]) / 2 for k in range(1, b+1)) / (b*f))
            computed.append(curve_metrics(counts, f)['normalized_auc_30'])
        np.testing.assert_allclose(computed, expected)
        self.assertAlmostEqual(np.mean(computed), np.mean(expected))

    def test_titles_follow_requested_order_with_independent_counters(self):
        groups = ['QAOA_12_1', 'IQFT_3', 'QAOA_12_4', 'IQFT_1']
        self.assertEqual(list(display_titles(groups).values()),
                         ['QAOA_1', 'IQFT_1', 'QAOA_2', 'IQFT_2'])
        self.assertEqual(display_titles(groups, {'QAOA_12_4': 'QAOA_4'})['QAOA_12_4'], 'QAOA_4')

    def test_rewards_and_penalties_use_separate_parameters(self):
        obj = QHDRT.__new__(QHDRT)
        obj.epsilon, obj.delta = .20, .10
        obj.subcluster_indices = {0: 0, 1: 1, 2: 2}
        obj.medoid_dist_matrix = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        obj.subcluster_probs = {0: .5, 1: .25, 2: .25}
        obj.update_probabilities(0, False)
        np.testing.assert_allclose(list(obj.subcluster_probs.values()), [.7, .15, .15])
        obj.subcluster_probs = {0: .5, 1: .25, 2: .25}
        obj.update_probabilities(0, True)
        np.testing.assert_allclose(list(obj.subcluster_probs.values()), [.4, .3, .3])

    def test_run_does_not_overwrite_parameters(self):
        class ReachedInitialization(Exception):
            pass
        class Probe(QHDRT):
            def _initialize_probabilities(self):
                if (self.epsilon, self.delta) != (.25, .01):
                    raise AssertionError('run() overwrote experiment parameters')
                raise ReachedInitialization
        obj = Probe.__new__(Probe)
        obj.epsilon, obj.delta = .25, .01
        obj.upper_bound, obj.iter_num, obj.seed = 4, 2, 123
        obj.classification_method, obj.distance_metric = 'type_clustering', 'hybrid'
        obj.n_clusters, obj.use_mixstate = 5, True
        with self.assertRaises(ReachedInitialization):
            obj.run()

    def test_metric_area_and_budget_rounding(self):
        result = curve_metrics([0, 1, 1, 2], 2)
        self.assertEqual(result['normalized_auc'], .375)
        self.assertEqual(result['first_failure'], 2)
        self.assertEqual(result['failures_10'], 0)
        self.assertEqual(result['recall_50'], .5)
        self.assertEqual(curve_metrics([1, 1, 1, 1], 1)['normalized_auc'], .875)
        self.assertEqual(curve_metrics([0, 0, 0, 1], 1)['normalized_auc'], .125)
        self.assertTrue(np.isnan(curve_metrics([0, 0, 0, 0], 0)['normalized_auc']))


if __name__ == '__main__':
    unittest.main()
