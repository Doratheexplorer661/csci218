import unittest
import tkinter as tk
from types import SimpleNamespace
from pathfinding.grid import Grid
from pathfinding.search import SearchSession
from pathfinding.dijkstra import DijkstraSession
from pathfinding.search_adapter import get_search_result
from pathfinding.visualization import EnvironmentUI, CELL_SIZE


class ContractTests(unittest.TestCase):
    def test_dispatches_to_team_dijkstra(self):
        session = SearchSession(Grid(4, 4), (0, 0), (3, 3), 'dijkstra')
        self.assertIsInstance(session, DijkstraSession)
        for algorithm in ['dijkstra', 'astar']:
            result, order = get_search_result(Grid(4, 4), (0, 0), (3, 3), algorithm)
            self.assertEqual(order, result['visited'])
            self.assertEqual(result['visited'], result['visited_order'])
            self.assertAlmostEqual(result['runtime_ms'], result['runtime'] * 1000)
            self.assertEqual(order[-1], (3, 3))


class UIIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.root.withdraw()
        self.ui = EnvironmentUI(self.root, rows=5, cols=7)

    def tearDown(self):
        if hasattr(self, 'ui'):
            self.ui._close()

    def click(self, cell, mode):
        self.ui.edit_mode.set(mode)
        self.ui.on_click(SimpleNamespace(x=cell[1]*CELL_SIZE+1, y=cell[0]*CELL_SIZE+1))

    def test_playback_and_settings(self):
        for algorithm in ['dijkstra', 'astar']:
            self.ui.algorithm.set(algorithm)
            self.ui.step()
            self.assertEqual(len(self.ui.playback.explored), 1)
            while not self.ui.playback.finished:
                self.ui.step()
            self.assertTrue(self.ui.result['found'])
            self.assertEqual(self.ui.playback.route, set(self.ui.result['path']))
        self.ui.movement.set('8')
        self.assertEqual(self.ui.heuristic.get(), 'diagonal')
        self.ui.step()
        self.ui.pause()
        self.assertIsNone(self.ui._timer)

    def test_live_block_replan_and_terrain(self):
        self.ui.toggle_robot()
        self.ui._stop_robot()
        position = self.ui.navigator.position
        self.ui.block_route()
        self.assertEqual(self.ui.navigator.replans, 2)
        self.assertEqual(self.ui.result['path'][0], position)
        self.assertTrue(all(c not in self.ui.grid.walls for c in self.ui.navigator.remaining))
        self.click((3, 1), 'terrain')
        self.assertEqual(self.ui.grid.weights[(3, 1)], 3)
        self.click((3, 1), 'erase')
        self.assertEqual(self.ui.grid.weights[(3, 1)], 1)
        self.click((2, 0), 'start')
        self.assertEqual(self.ui.start, (2, 0))
        self.assertIsNone(self.ui.navigator)

    def test_no_path_and_recovery(self):
        self.ui.grid = Grid(1, 7)
        self.ui.start, self.ui.goal = (0, 0), (0, 6)
        self.ui.toggle_robot()
        self.ui._stop_robot()
        self.click((0, 3), 'wall')
        self.assertFalse(self.ui.result['found'])
        self.assertFalse(self.ui.navigator.tick())
        self.click((0, 3), 'erase')
        self.assertTrue(self.ui.result['found'])
        for _ in range(10):
            self.ui.navigator.tick()
        self.assertEqual(self.ui.navigator.position, self.ui.goal)

    def test_seeded_generation_and_protection(self):
        self.ui.generate_map()
        old = self.ui.grid.clone()
        self.ui.generate_map()
        self.assertEqual(old, self.ui.grid)
        self.click(self.ui.start, 'wall')
        self.assertTrue(self.ui.grid.free(self.ui.start))
