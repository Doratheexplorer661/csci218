import math
import os
import tempfile
import unittest
from unittest import mock

from pathfinding import analysis, benchmark
from pathfinding.experiments import (GridConfig, derive_seed, generate_instance,
                                     generate_instances, grid_fingerprint,
                                     is_solvable)
from pathfinding.grid import Grid
from pathfinding.search import SearchSession


def snapshot(instance):
    g = instance.grid
    return (g.rows, g.cols, frozenset(g.walls), tuple(sorted(g.weights.items())),
            instance.start, instance.goal)


def fake_row(found=True, cost=1.0, nodes=10, runtime=1.0, algorithm='dijkstra',
             heuristic='none', movement=4, optimal=True, solvable=True, seed=1):
    return dict(seed=seed, trial=0, rows=5, cols=5, density=0.2, terrain_fraction=0.0,
                n_walls=3, movement=movement, algorithm=algorithm, heuristic=heuristic,
                solvable=solvable, found=found, cost=cost if found else None,
                nodes_expanded=nodes, runtime_ms=runtime, optimal=optimal,
                grid_hash='abc')


class TestGridConfig(unittest.TestCase):
    def test_invalid_values_rejected(self):
        for kwargs in (dict(rows=0, cols=5), dict(rows=5, cols=-1),
                       dict(rows=5, cols=5, density=1.5),
                       dict(rows=5, cols=5, movement=6),
                       dict(rows=5, cols=5, terrain_fraction=0.5, terrain_weights=(0.5,)),
                       dict(rows=5, cols=5, endpoints='middle')):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                GridConfig(**kwargs)


class TestGenerator(unittest.TestCase):
    def test_same_seed_same_grid(self):
        cfg = GridConfig(15, 15, density=0.3, terrain_fraction=0.2)
        a, b = generate_instance(cfg, 7), generate_instance(cfg, 7)
        self.assertEqual(snapshot(a), snapshot(b))
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_different_seeds_differ(self):
        cfg = GridConfig(15, 15, density=0.3)
        prints = {generate_instance(cfg, s).fingerprint for s in range(5)}
        self.assertGreater(len(prints), 1)

    def test_global_random_state_untouched(self):
        import random
        random.seed(123)
        expected = random.random()
        random.seed(123)
        generate_instance(GridConfig(10, 10, density=0.3), 5)
        self.assertEqual(random.random(), expected)

    def test_start_and_goal_never_blocked_even_at_full_density(self):
        for endpoints in ('corners', 'random'):
            for seed in range(10):
                cfg = GridConfig(6, 6, density=1.0, terrain_fraction=0.5, endpoints=endpoints)
                inst = generate_instance(cfg, seed)
                self.assertNotIn(inst.start, inst.grid.walls)
                self.assertNotIn(inst.goal, inst.grid.walls)
                self.assertNotIn(inst.start, inst.grid.weights)
                self.assertNotIn(inst.goal, inst.grid.weights)
                self.assertNotEqual(inst.start, inst.goal)

    def test_corner_endpoints(self):
        inst = generate_instance(GridConfig(4, 7), 1)
        self.assertEqual((inst.start, inst.goal), ((0, 0), (3, 6)))

    def test_density_extremes(self):
        self.assertEqual(len(generate_instance(GridConfig(8, 8, density=0.0), 1).grid.walls), 0)
        full = generate_instance(GridConfig(8, 8, density=1.0), 1)
        self.assertEqual(len(full.grid.walls), 8 * 8 - 2)

    def test_density_is_close_to_requested(self):
        inst = generate_instance(GridConfig(60, 60, density=0.25), 3)
        realised = len(inst.grid.walls) / (60 * 60 - 2)
        self.assertAlmostEqual(realised, 0.25, delta=0.03)

    def test_terrain_weights_valid_and_not_on_walls(self):
        cfg = GridConfig(20, 20, density=0.2, terrain_fraction=0.4, terrain_weights=(2.0, 4.0))
        inst = generate_instance(cfg, 9)
        self.assertTrue(inst.grid.weights)
        for cell, w in inst.grid.weights.items():
            self.assertIn(w, (2.0, 4.0))
            self.assertNotIn(cell, inst.grid.walls)

    def test_no_terrain_by_default(self):
        self.assertEqual(generate_instance(GridConfig(10, 10), 1).grid.weights, {})

    def test_solvable_only_returns_solvable_and_is_deterministic(self):
        cfg = GridConfig(12, 12, density=0.4)
        a = generate_instance(cfg, 11, solvable_only=True)
        b = generate_instance(cfg, 11, solvable_only=True)
        self.assertTrue(a.solvable)
        self.assertTrue(is_solvable(a.grid, a.start, a.goal, 4))
        self.assertEqual(a.attempts, b.attempts)
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_solvable_only_raises_when_impossible(self):
        with self.assertRaises(RuntimeError):
            generate_instance(GridConfig(5, 5, density=1.0), 1, solvable_only=True,
                              max_attempts=5)

    def test_unsolvable_cases_are_recorded(self):
        inst = generate_instance(GridConfig(5, 5, density=1.0), 1, solvable_only=False)
        self.assertFalse(inst.solvable)

    def test_one_by_one_grid(self):
        inst = generate_instance(GridConfig(1, 1, density=0.9), 1)
        self.assertEqual(inst.start, inst.goal)
        self.assertTrue(inst.solvable)

    def test_generate_instances_reproducible_and_distinct(self):
        cfg = GridConfig(10, 10, density=0.3)
        first = [i.fingerprint for i in generate_instances(cfg, 218, 5)]
        again = [i.fingerprint for i in generate_instances(cfg, 218, 5)]
        other = [i.fingerprint for i in generate_instances(cfg, 219, 5)]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertGreater(len(set(first)), 1)

    def test_derive_seed_stable_and_condition_sensitive(self):
        self.assertEqual(derive_seed(1, 'a', 2), derive_seed(1, 'a', 2))
        self.assertNotEqual(derive_seed(1, 'a', 2), derive_seed(1, 'a', 3))


class TestSolvabilityOracle(unittest.TestCase):
    def test_open_and_walled(self):
        g = Grid(3, 3)
        self.assertTrue(is_solvable(g, (0, 0), (2, 2)))
        g.set_wall((1, 2))
        g.set_wall((2, 1))
        self.assertFalse(is_solvable(g, (0, 0), (2, 2)))

    def test_diagonal_corner_cutting_not_allowed(self):
        g = Grid(2, 2)
        g.set_wall((0, 1))
        g.set_wall((1, 0))
        self.assertFalse(is_solvable(g, (0, 0), (1, 1), movement=8))

    def test_blocked_or_outside_endpoints(self):
        g = Grid(3, 3)
        g.set_wall((2, 2))
        self.assertFalse(is_solvable(g, (0, 0), (2, 2)))
        self.assertFalse(is_solvable(g, (0, 0), (9, 9)))


class TestBenchmarkSelection(unittest.TestCase):
    def test_manhattan_rejected_for_eight_way(self):
        with self.assertRaises(ValueError):
            benchmark.check_combination('astar', 'manhattan', 8)
        self.assertNotIn(('astar', 'manhattan'), benchmark.algorithms_for(8))
        self.assertIn(('astar', 'manhattan'), benchmark.algorithms_for(4))

    def test_eight_way_plan_contents(self):
        self.assertEqual(benchmark.algorithms_for(8),
                         [('dijkstra', None), ('astar', 'euclidean'), ('astar', 'diagonal')])

    def test_run_single_refuses_manhattan_eight_way(self):
        inst = generate_instance(GridConfig(5, 5, 0.1), 1)
        with self.assertRaises(ValueError):
            benchmark.run_single(inst, 'astar', 'manhattan', 8)

    def test_search_session_itself_also_rejects_it(self):
        # Integration check: the teammates' engine agrees with our rule.
        with self.assertRaises(ValueError):
            SearchSession(Grid(4, 4), (0, 0), (3, 3), 'astar', 'manhattan', 8)

    def test_unknown_names_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.check_combination('bfs', None, 4)
        with self.assertRaises(ValueError):
            benchmark.check_combination('astar', 'chebyshev', 4)


class TestFairness(unittest.TestCase):
    def test_every_algorithm_receives_identical_grid(self):
        cfg = benchmark.ExperimentConfig(sizes=((10, 10),), densities=(0.3,),
                                         movements=(4, 8), trials=3, runtime_repeats=1)
        seen = []
        real = SearchSession

        def spy(grid, start, goal, algorithm, heuristic, movement):
            seen.append((grid_fingerprint(grid, start, goal), algorithm, heuristic, movement))
            return real(grid, start, goal, algorithm, heuristic, movement)

        with mock.patch.object(benchmark, 'SearchSession', side_effect=spy):
            rows = benchmark.run_benchmark(cfg)
        self.assertEqual(len(seen), len(rows))
        by_trial = {}
        for r in rows:
            by_trial.setdefault(r['trial'], set()).add(r['grid_hash'])
        self.assertTrue(all(len(h) == 1 for h in by_trial.values()))
        self.assertEqual(len({f for f, *_ in seen}), 3)  # 3 trials -> 3 distinct grids
        for fp, *_ in seen:
            self.assertIn(fp, {h for hs in by_trial.values() for h in hs})

    def test_run_does_not_mutate_the_instance(self):
        inst = generate_instance(GridConfig(10, 10, 0.3, terrain_fraction=0.2), 4)
        before = snapshot(inst)
        for algorithm, heuristic in benchmark.algorithms_for(4):
            benchmark.run_single(inst, algorithm, heuristic, 4)
        self.assertEqual(before, snapshot(inst))

    def test_both_movements_share_one_grid_per_trial(self):
        cfg = benchmark.ExperimentConfig(sizes=((8, 8),), densities=(0.2,),
                                         movements=(4, 8), trials=2, runtime_repeats=1)
        rows = benchmark.run_benchmark(cfg)
        for trial in (0, 1):
            hashes = {r['grid_hash'] for r in rows if r['trial'] == trial}
            self.assertEqual(len(hashes), 1)

    def test_row_count(self):
        cfg = benchmark.ExperimentConfig(sizes=((6, 6),), densities=(0.1, 0.2),
                                         movements=(4, 8), trials=3, runtime_repeats=1)
        rows = benchmark.run_benchmark(cfg)
        # per trial: 4 algorithms for 4-way + 3 for 8-way = 7
        self.assertEqual(len(rows), 2 * 3 * 7)


class TestCorrectnessOfRuns(unittest.TestCase):
    def test_astar_cost_equals_dijkstra_cost(self):
        for movement in (4, 8):
            for seed in range(15):
                inst = generate_instance(GridConfig(12, 12, 0.25, terrain_fraction=0.3), seed,
                                         solvable_only=True)
                costs = [benchmark.run_single(inst, a, h, movement)['cost']
                         for a, h in benchmark.algorithms_for(movement)]
                for c in costs:
                    self.assertAlmostEqual(c, costs[0], places=9)

    def test_optimal_column_and_consistency_clean(self):
        cfg = benchmark.ExperimentConfig(sizes=((10, 10),), densities=(0.2,), trials=5,
                                         runtime_repeats=1, terrain_fraction=0.2)
        rows = benchmark.run_benchmark(cfg)
        self.assertTrue(all(r['optimal'] for r in rows))
        self.assertEqual(analysis.check_consistency(rows), [])

    def test_no_path_row_has_no_cost_and_real_effort(self):
        inst = generate_instance(GridConfig(6, 6, density=1.0), 1)
        row = benchmark.run_single(inst, 'dijkstra', None, 4)
        self.assertFalse(row['found'])
        self.assertIsNone(row['cost'])
        self.assertFalse(row['solvable'])
        self.assertGreaterEqual(row['nodes_expanded'], 1)

    def test_unsolvable_instances_are_recorded_when_allowed(self):
        cfg = benchmark.ExperimentConfig(sizes=((6, 6),), densities=(1.0,), movements=(4,),
                                         trials=2, solvable_only=False, runtime_repeats=1)
        rows = benchmark.run_benchmark(cfg)
        self.assertTrue(rows)
        self.assertTrue(all(not r['found'] and not r['solvable'] for r in rows))
        self.assertTrue(all(r['optimal'] for r in rows))  # all agree: no path

    def test_deterministic_metrics_repeat_exactly(self):
        cfg = benchmark.ExperimentConfig(sizes=((10, 10),), densities=(0.3,), trials=3,
                                         runtime_repeats=1)
        a, b = benchmark.run_benchmark(cfg), benchmark.run_benchmark(cfg)
        key = lambda r: (r['seed'], r['movement'], r['algorithm'], r['heuristic'],
                         r['found'], r['cost'], r['nodes_expanded'], r['grid_hash'])
        self.assertEqual([key(r) for r in a], [key(r) for r in b])

    def test_runtime_repeats_uses_median(self):
        inst = generate_instance(GridConfig(5, 5, 0.0), 1)
        times = iter([5.0, 1.0, 3.0])
        real = SearchSession

        class Timed:
            def __init__(self, *args):
                self.inner = real(*args)

            def run(self):
                result = dict(self.inner.run())
                result['runtime_ms'] = next(times)
                return result

        with mock.patch.object(benchmark, 'SearchSession', Timed):
            row = benchmark.run_single(inst, 'dijkstra', None, 4, runtime_repeats=3)
        self.assertEqual(row['runtime_ms'], 3.0)


class TestMetrics(unittest.TestCase):
    def test_mean_std_known_values(self):
        mean, std = analysis.mean_std([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(std, 2.13808993529939)  # sample std (n-1)

    def test_mean_std_edge_cases(self):
        self.assertTrue(all(math.isnan(v) for v in analysis.mean_std([])))
        mean, std = analysis.mean_std([3.0])
        self.assertEqual(mean, 3.0)
        self.assertTrue(math.isnan(std))

    def test_infinite_and_none_values_are_ignored(self):
        mean, _ = analysis.mean_std([1.0, math.inf, None, 3.0])
        self.assertEqual(mean, 2.0)

    def test_failed_runs_excluded_from_means_but_counted_in_success_rate(self):
        rows = [fake_row(cost=10, nodes=100, runtime=2.0),
                fake_row(cost=20, nodes=300, runtime=4.0),
                fake_row(found=False, nodes=999, runtime=99.0)]
        s = analysis.summarise_group(rows)
        self.assertEqual((s['n_trials'], s['n_found']), (3, 2))
        self.assertAlmostEqual(s['success_rate'], 2 / 3)
        self.assertEqual(s['cost_mean'], 15.0)
        self.assertEqual(s['nodes_expanded_mean'], 200.0)
        self.assertEqual(s['runtime_ms_mean'], 3.0)
        self.assertTrue(math.isfinite(s['cost_mean']))

    def test_all_failed_group(self):
        s = analysis.summarise_group([fake_row(found=False), fake_row(found=False)])
        self.assertEqual(s['success_rate'], 0.0)
        self.assertTrue(math.isnan(s['cost_mean']))
        self.assertTrue(math.isnan(s['cost_std']))

    def test_summarise_groups_and_relative_nodes(self):
        rows = [fake_row(nodes=100, algorithm='dijkstra'),
                fake_row(nodes=100, algorithm='dijkstra', seed=2),
                fake_row(nodes=50, algorithm='astar', heuristic='diagonal'),
                fake_row(nodes=50, algorithm='astar', heuristic='diagonal', seed=2)]
        summaries = {(s['algorithm'], s['heuristic']): s for s in analysis.summarise(rows)}
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[('dijkstra', 'none')]['nodes_vs_dijkstra'], 1.0)
        self.assertEqual(summaries[('astar', 'diagonal')]['nodes_vs_dijkstra'], 0.5)

    def test_consistency_check_flags_problems(self):
        self.assertEqual(analysis.check_consistency([fake_row()]), [])
        bad_cost = analysis.check_consistency([fake_row(optimal=False)])
        self.assertEqual(len(bad_cost), 1)
        bad_found = analysis.check_consistency([fake_row(found=True, solvable=False)])
        self.assertEqual(len(bad_found), 1)


class TestOutputFiles(unittest.TestCase):
    def test_csv_round_trip_charts_and_summary(self):
        cfg = benchmark.ExperimentConfig(sizes=((8, 8),), densities=(0.1, 0.3),
                                         movements=(4, 8), trials=4, runtime_repeats=1,
                                         solvable_only=False)
        rows = benchmark.run_benchmark(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, 'raw_results.csv')
            benchmark.write_raw_csv(rows, raw)
            loaded = analysis.load_raw_csv(raw)
            self.assertEqual(len(loaded), len(rows))
            for a, b in zip(rows, loaded):
                self.assertEqual(a['found'], b['found'])
                self.assertEqual(a['cost'], b['cost'])
                self.assertEqual(a['nodes_expanded'], b['nodes_expanded'])
            _, charts, problems = analysis.run_analysis(loaded, tmp)
            self.assertEqual(problems, [])
            self.assertTrue(os.path.exists(os.path.join(tmp, 'summary_stats.csv')))
            self.assertEqual(len(charts), 6)  # 3 metrics x 2 movement modes
            for path in charts:
                self.assertGreater(os.path.getsize(path), 1000)
                with open(path, 'rb') as fh:
                    self.assertEqual(fh.read(8), b'\x89PNG\r\n\x1a\n')

    def test_failed_costs_written_as_empty_not_inf(self):
        row = fake_row(found=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'r.csv')
            benchmark.write_raw_csv([row], path)
            with open(path) as fh:
                text = fh.read()
        self.assertNotIn('inf', text.lower())


if __name__ == '__main__':
    unittest.main()
