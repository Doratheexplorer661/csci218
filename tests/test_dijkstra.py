import math
import unittest

from pathfinding.grid import Grid
from pathfinding.dijkstra import dijkstra


def make_grid(rows, cols, walls=(), weights=None):
    grid = Grid(rows, cols)
    for wall in walls:
        grid.set_wall(wall)
    if weights:
        grid.weights.update(weights)
    return grid


def is_valid_path(grid, path, movement=4):
    """Every cell passable and every step a legal move on the grid."""
    if not all(grid.passable(cell) for cell in path):
        return False
    return all(b in dict(grid.neighbors(a, movement))
               for a, b in zip(path, path[1:]))


class TestDijkstra(unittest.TestCase):
    def test_open_grid(self):
        grid = make_grid(5, 5)
        r = dijkstra(grid, (0, 0), (4, 4))
        self.assertTrue(r['found'])
        self.assertEqual(r['cost'], 8)
        self.assertEqual(r['path'][0], (0, 0))
        self.assertEqual(r['path'][-1], (4, 4))
        self.assertTrue(is_valid_path(grid, r['path']))
        self.assertEqual(len(r['path']) - 1, r['cost'])

    def test_start_equals_goal(self):
        r = dijkstra(make_grid(3, 3), (1, 1), (1, 1))
        self.assertTrue(r['found'])
        self.assertEqual(r['path'], [(1, 1)])
        self.assertEqual(r['cost'], 0)

    def test_no_path_goal_walled_off(self):
        grid = make_grid(3, 3, walls=[(1, 2), (2, 1)])
        r = dijkstra(grid, (0, 0), (2, 2))
        self.assertFalse(r['found'])
        self.assertEqual(r['path'], [])
        self.assertEqual(r['cost'], math.inf)
        self.assertGreater(r['nodes_expanded'], 0)

    def test_start_on_wall(self):
        grid = make_grid(3, 3, walls=[(0, 0)])
        r = dijkstra(grid, (0, 0), (2, 2))
        self.assertFalse(r['found'])
        self.assertEqual(r['path'], [])

    def test_goal_on_wall(self):
        grid = make_grid(3, 3, walls=[(2, 2)])
        self.assertFalse(dijkstra(grid, (0, 0), (2, 2))['found'])

    def test_out_of_bounds(self):
        grid = make_grid(3, 3)
        self.assertFalse(dijkstra(grid, (-1, 0), (2, 2))['found'])
        self.assertFalse(dijkstra(grid, (0, 0), (9, 9))['found'])

    def test_one_by_one_grid(self):
        r = dijkstra(make_grid(1, 1), (0, 0), (0, 0))
        self.assertTrue(r['found'])
        self.assertEqual(r['path'], [(0, 0)])

    def test_single_row_corridor(self):
        grid = make_grid(1, 5)
        r = dijkstra(grid, (0, 0), (0, 4))
        self.assertEqual(r['cost'], 4)
        self.assertEqual(r['path'], [(0, i) for i in range(5)])

    def test_corridor_blocked_in_middle(self):
        grid = make_grid(1, 5, walls=[(0, 2)])
        self.assertFalse(dijkstra(grid, (0, 0), (0, 4))['found'])

    def test_detour_around_wall(self):
        grid = make_grid(3, 3, walls=[(1, 0), (1, 1)])
        r = dijkstra(grid, (0, 0), (2, 0))
        self.assertEqual(r['cost'], 6)
        self.assertTrue(is_valid_path(grid, r['path']))

    def test_equal_cost_paths_check_cost_only(self):
        grid = make_grid(3, 3)
        r = dijkstra(grid, (0, 0), (2, 2))
        self.assertEqual(r['cost'], 4)
        self.assertTrue(is_valid_path(grid, r['path']))

    def test_terrain_weights_avoid_expensive_cell(self):
        grid = make_grid(2, 3, weights={(1, 1): 9})
        r = dijkstra(grid, (1, 0), (1, 2))
        self.assertEqual(r['cost'], 4)
        self.assertNotIn((1, 1), r['path'])

    def test_eight_way_movement(self):
        grid = make_grid(4, 4)
        r = dijkstra(grid, (0, 0), (3, 3), movement=8)
        self.assertAlmostEqual(r['cost'], 3 * math.sqrt(2))
        self.assertEqual(len(r['path']), 4)
        self.assertTrue(is_valid_path(grid, r['path'], movement=8))

    def test_no_diagonal_corner_cutting(self):
        # Walls on both sides of the diagonal block it in the shared Grid
        grid = make_grid(2, 2, walls=[(0, 1), (1, 0)])
        self.assertFalse(dijkstra(grid, (0, 0), (1, 1), movement=8)['found'])

    def test_cost_matches_sum_of_steps(self):
        grid = make_grid(3, 4, walls=[(1, 1), (1, 2)])
        r = dijkstra(grid, (0, 0), (0, 3))
        self.assertEqual(r['cost'], len(r['path']) - 1)

    def test_visited_order_starts_at_start_and_has_no_repeats(self):
        grid = make_grid(4, 4)
        r = dijkstra(grid, (0, 0), (3, 3))
        self.assertEqual(r['visited_order'][0], (0, 0))
        self.assertEqual(len(r['visited_order']), len(set(r['visited_order'])))
        self.assertEqual(r['nodes_expanded'], len(r['visited_order']))

    def test_grid_not_modified(self):
        grid = make_grid(3, 3, walls=[(1, 1)])
        walls_before = set(grid.walls)
        dijkstra(grid, (0, 0), (2, 2))
        self.assertEqual(grid.walls, walls_before)


if __name__ == '__main__':
    unittest.main()
