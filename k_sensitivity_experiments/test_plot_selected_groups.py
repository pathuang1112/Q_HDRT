"""Checks for combined plot data selection, statistics, titles and axes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from plot_selected_groups import (BUDGET_LABEL, CURVE_LABEL, build_display_titles,
                                  draw_pair, load_group, plot_selected, source_root)


def write_fixture(root, group, k=3):
    folder = root / 'raw' / group
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(subject={'group': group}, k=k, n_cases=10, total_failures=2,
                runs=[dict(seed=10+i, cumulative_failures=counts) for i, counts in enumerate([
                    [0, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                    [0, 0, 1, 1, 2, 2, 2, 2, 2, 2]])])
    (folder / f'k_{k}.json').write_text(json.dumps(data), encoding='utf-8')
    return folder / f'k_{k}.json'


class SelectedPlotTests(unittest.TestCase):
    def test_mapping_preserves_selection_and_reserves_qaoa_one(self):
        groups = ['qft_8_5', 'QAOA_12_2', 'adder_n7_3', 'QAOA_12_1', 'qft_8_2']
        titles = build_display_titles(groups)
        self.assertEqual(list(titles), groups)
        self.assertEqual(list(titles.values()), ['QFT_1', 'QAOA_2', 'Adder_1', 'QAOA_1', 'QFT_2'])

    def test_source_routing(self):
        self.assertEqual(source_root('QAOA_12_1', 'qaoa', 'others'), Path('qaoa'))
        self.assertEqual(source_root('qft_8_2', 'qaoa', 'others'), Path('others'))

    def test_statistics_and_requested_axes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, 'QAOA_12_1')
            records = load_group('QAOA_12_1', root)
            self.assertEqual(records[0]['budget_mean'], 1)
            self.assertEqual(records[0]['budget_30'], 3)
            np.testing.assert_allclose(records[0]['curve_mean'], [0, 0, .5, 1, 1.5, 2, 2, 2, 2, 2, 2])
            fig, pair = plt.subplots(1, 2)
            draw_pair(pair[0], pair[1], records, 'QAOA_1', {3: 'blue'})
            self.assertEqual(pair[0].get_ylim()[0], 0)
            self.assertEqual(pair[0].get_ylabel().replace('\n', ' '), BUDGET_LABEL)
            self.assertEqual(pair[1].get_ylabel(), CURVE_LABEL)
            self.assertEqual(pair[1].get_xlim(), (0, 100))
            self.assertEqual(pair[0].get_title(), 'QAOA_1')
            plt.close(fig)

    def test_missing_k_and_invalid_curves_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_fixture(root, 'qft_8_2')
            with self.assertRaisesRegex(ValueError, 'missing k'):
                load_group('qft_8_2', root, [1, 3])
            data = json.loads(path.read_text(encoding='utf-8'))
            data['runs'][0]['cumulative_failures'][2] = 10
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid cumulative'):
                load_group('qft_8_2', root)

    def test_four_panels_per_row_in_selection_order_and_odd_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            qaoa, other = root / 'qaoa', root / 'other'
            groups = ['QAOA_12_1', 'qft_8_2', 'adder_n7_3']
            write_fixture(qaoa, groups[0])
            for group in groups[1:]:
                write_fixture(other, group)
            captures = {}

            def capture(fig, output, name, dpi):
                captures[name] = [(ax.get_title(), ax.get_visible()) for ax in fig.axes]
                plt.close(fig)

            with patch('plot_selected_groups.save_figure', side_effect=capture):
                plot_selected(groups, qaoa, other, root / 'plots', name='combined')
            self.assertEqual(captures['combined'], [
                ('QAOA_1', True), ('QAOA_1', True), ('QFT_1', True), ('QFT_1', True),
                ('Adder_1', True), ('Adder_1', True), ('', False), ('', False)])
            report = json.loads((root / 'plots' / 'combined_sources.json').read_text(encoding='utf-8'))
            self.assertEqual([r['group'] for r in report['groups']], groups)
            self.assertEqual(report['groups'][2]['columns'], [1, 2])


if __name__ == '__main__':
    unittest.main()
