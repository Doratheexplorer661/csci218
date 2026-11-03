import unittest
from pathfinding import Grid
from pathfinding.dynamic import Navigator


class DynamicTests(unittest.TestCase):
    def test_replan_from_current_cell(self):
        n = Navigator(Grid(5, 7), (2, 0), (2, 6))
        n.tick()
        position = n.position
        n.edit((2, 3), wall=True)
        self.assertEqual(n.result['path'][0], position)
        self.assertNotIn((2, 3), n.result['path'])
        self.assertEqual(n.replans, 2)
        for _ in range(40):
            n.tick()
        self.assertEqual(n.position, n.goal)
        self.assertEqual(n.travel_cost, 8)

    def test_no_path_then_recovery(self):
        n = Navigator(Grid(1, 5), (0, 0), (0, 4))
        n.edit((0, 2), wall=True)
        self.assertFalse(n.result['found'])
        self.assertFalse(n.tick())
        n.edit((0, 2), wall=False)
        self.assertTrue(n.result['found'])
        for _ in range(4):
            self.assertTrue(n.tick())
        self.assertFalse(n.tick())

    def test_weight_change_and_protection(self):
        n = Navigator(Grid(3, 5), (1, 0), (1, 4))
        n.edit((1, 2), weight=10)
        self.assertNotIn((1, 2), n.result['path'])
        n.edit((1, 2), weight=1)
        self.assertEqual(n.result['cost'], 4)
        for p in (n.position, n.goal):
            with self.assertRaises(ValueError):
                n.edit(p, wall=True)
        with self.assertRaises(ValueError):
            n.edit((1, 2), weight=0)

    def test_diagonal_side_cell_change(self):
        n = Navigator(Grid(3, 3), (0, 0), (2, 2), heuristic='diagonal', movement=8)
        n.edit((0, 1), wall=True)
        self.assertNotEqual(n.remaining[0], (1, 1))
