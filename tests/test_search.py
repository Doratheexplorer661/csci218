import math
import random
import unittest
from pathfinding import Grid, random_grid, dijkstra, astar, SearchSession
from pathfinding.heuristics import HEURISTICS


def reference_cost(grid, start, goal, movement):
    """Independent Bellman-Ford oracle, with no heap or heuristic."""
    cells = [(r, c) for r in range(grid.rows) for c in range(grid.cols) if grid.free((r, c))]
    dist = {p: math.inf for p in cells}
    dist[start] = 0.0
    for _ in range(len(cells)-1):
        changed = False
        for p in cells:
            for q, cost in grid.neighbors(p, movement):
                if dist[p]+cost < dist[q]:
                    dist[q] = dist[p]+cost
                    changed = True
        if not changed:
            break
    return dist[goal]


class SearchTests(unittest.TestCase):
    def check_path(self, grid, result, start, goal, movement):
        self.assertEqual(result['nodes_expanded'], len(result['visited']))
        self.assertEqual(len(result['visited']), len(set(result['visited'])))
        self.assertGreaterEqual(result['runtime_ms'], 0)
        if result['found']:
            self.assertEqual(result['path'][0], start)
            self.assertEqual(result['path'][-1], goal)
            cost = 0.0
            for p, q in zip(result['path'], result['path'][1:]):
                self.assertIn(q, dict(grid.neighbors(p, movement)))
                cost += dict(grid.neighbors(p, movement))[q]
            self.assertAlmostEqual(result['cost'], cost)
        else:
            self.assertEqual(result['path'], [])
            self.assertEqual(result['cost'], math.inf)

    def test_open_grid(self):
        for fn in (dijkstra, astar):
            z = fn(Grid(5, 7), (0, 0), (4, 6))
            self.assertEqual(z['cost'], 10)
            self.check_path(Grid(5, 7), z, (0, 0), (4, 6), 4)

    def test_no_path(self):
        g = Grid(3, 3, {(1, c) for c in range(3)})
        for fn in (dijkstra, astar):
            z = fn(g, (0, 0), (2, 2))
            self.assertFalse(z['found'])
            self.check_path(g, z, (0, 0), (2, 2), 4)

    def test_single_cell_start_equals_goal(self):
        for fn in (dijkstra, astar):
            z = fn(Grid(1, 1), (0, 0), (0, 0))
            self.assertEqual(z['path'], [(0, 0)])
            self.assertEqual(z['cost'], 0)
            self.assertEqual(z['nodes_expanded'], 1)

    def test_thin_grid(self):
        self.assertEqual(astar(Grid(1, 8), (0, 0), (0, 7))['cost'], 7)
        self.assertEqual(dijkstra(Grid(8, 1), (0, 0), (7, 0))['cost'], 7)

    def test_invalid_inputs(self):
        for matrix in ([], [[]], [[0], [0, 1]], [[2]]):
            with self.assertRaises(ValueError):
                astar(matrix, (0, 0), (0, 0))
        for start, goal in [((-1, 0), (1, 1)), ((0, 0), (2, 0)), ((0, 1), (1, 1))]:
            with self.assertRaises(ValueError):
                dijkstra(Grid(2, 2, {(0, 1)}), start, goal)
        for weight in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Grid(2, 2, weights={(0, 0): weight})
        for dims in [(0, 1), (1, -1), (1.5, 2)]:
            with self.assertRaises(ValueError):
                Grid(*dims)
        with self.assertRaises(ValueError):
            astar(Grid(2, 2), (0, 0), (1, 1), movement=8)
        with self.assertRaises(ValueError):
            astar(Grid(2, 2), (0, 0), (1, 1), 'unknown')
        with self.assertRaises(ValueError):
            dijkstra(Grid(2, 2), (0, 0), (1, 1), movement=6)

    def test_legacy_api(self):
        from astar import astar as legacy_a
        from dijkstra import dijkstra as legacy_d
        g = [[0, 1, 0], [0, 0, 0]]
        self.assertEqual(legacy_a(g, (0, 0), (0, 2))['cost'], 4)
        self.assertEqual(legacy_a(g, (0, 0), (0, 2)).keys(), legacy_d(g, (0, 0), (0, 2)).keys())

    def test_diagonal_and_corner(self):
        self.assertAlmostEqual(astar(Grid(3, 3), (0, 0), (2, 2), 'diagonal', 8)['cost'], 2*math.sqrt(2))
        g = Grid(2, 2, {(0, 1), (1, 0)})
        self.assertFalse(astar(g, (0, 0), (1, 1), 'diagonal', 8)['found'])
        self.assertEqual(dijkstra(Grid(2, 2, {(0, 1)}), (0, 0), (1, 1), 8)['cost'], 2)

    def test_weighted_detour(self):
        g = Grid(3, 5, weights={(1, c): 9 for c in range(1, 4)})
        for fn in (dijkstra, astar):
            z = fn(g, (1, 0), (1, 4))
            self.assertEqual(z['cost'], 6)
            self.check_path(g, z, (1, 0), (1, 4), 4)

    def test_oracle_many_seeded_grids(self):
        for movement in (4, 8):
            names = list(HEURISTICS) if movement == 4 else ['euclidean', 'diagonal']
            for seed in range(25):
                g = random_grid(6, 6, .25, seed, weighted=True)
                expected = reference_cost(g, (0, 0), (5, 5), movement)
                for name in ['dijkstra'] + names:
                    with self.subTest(movement=movement, seed=seed, algorithm=name):
                        z = dijkstra(g, (0, 0), (5, 5), movement) if name == 'dijkstra' else astar(g, (0, 0), (5, 5), name, movement)
                        self.assertAlmostEqual(z['cost'], expected)
                        self.check_path(g, z, (0, 0), (5, 5), movement)

    def test_heuristic_consistency(self):
        for movement in (4, 8):
            g = random_grid(6, 6, .1, 17, weighted=True)
            for name, h in HEURISTICS.items():
                if movement == 8 and name == 'manhattan':
                    continue
                for r in range(6):
                    for c in range(6):
                        p = r, c
                        if g.free(p):
                            for q, cost in g.neighbors(p, movement):
                                self.assertLessEqual(h(p, (5, 5)), cost+h(q, (5, 5))+1e-12)

    def test_incremental_snapshot(self):
        g = Grid(4, 4)
        s = SearchSession(g, (0, 0), (3, 3))
        g.walls.add((3, 3))
        while s.result is None:
            before = len(s.visited)
            s.step()
            self.assertLessEqual(len(s.visited)-before, 1)
        self.assertTrue(s.result['found'])
        self.assertIsNone(s.step())

    def test_random_reproducibility(self):
        random.seed(999)
        a = random_grid(20, 20, .2, 218, True)
        random.random()
        self.assertEqual(a, random_grid(20, 20, .2, 218, True))
        self.assertNotEqual(a, random_grid(20, 20, .2, 219, True))
        g = random_grid(3, 3, 1)
        self.assertEqual(len(g.walls), 7)
        with self.assertRaises(ValueError):
            random_grid(density=1.1)
