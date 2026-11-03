"""Shared incremental interface dispatching to the team's Dijkstra or A* code."""
import heapq
import itertools
import math
from time import perf_counter
from .grid import Grid
from .heuristics import HEURISTICS


class SearchSession:
    """step() expands at most one node. A private snapshot isolates live edits.

    Runtime accumulates computation only, excluding time between UI frames.
    Result is None until done, then both algorithms expose the same dict API.
    """
    def __new__(cls, grid, start, goal, algorithm='astar', heuristic='manhattan', movement=4):
        if algorithm == 'dijkstra':
            from .dijkstra import DijkstraSession
            if heuristic not in HEURISTICS:
                raise ValueError('Unknown heuristic')
            return DijkstraSession(grid, start, goal, movement)
        return super().__new__(cls)

    def __init__(self, grid, start, goal, algorithm='astar', heuristic='manhattan', movement=4):
        if algorithm not in ('astar', 'dijkstra') or movement not in (4, 8):
            raise ValueError('Unknown algorithm or movement')
        if heuristic not in HEURISTICS:
            raise ValueError('Unknown heuristic')
        if algorithm == 'astar' and movement == 8 and heuristic == 'manhattan':
            raise ValueError('Manhattan is not admissible for sqrt(2) diagonal moves; use diagonal or euclidean')
        self.grid = grid.clone() if isinstance(grid, Grid) else Grid.from_matrix(grid)
        # Revalidate mutable terrain before taking a search snapshot.
        self.grid.__post_init__()
        if not self.grid.free(start) or not self.grid.free(goal):
            raise ValueError('Start and goal must be in bounds and unblocked')
        self.start, self.goal, self.movement = start, goal, movement
        self.h = HEURISTICS[heuristic] if algorithm == 'astar' else lambda a, b: 0.0
        self.counter = itertools.count()
        self.heap = [(self.h(start, goal), 0.0, next(self.counter), start)]
        self.g, self.parent = {start: 0.0}, {}
        self.closed, self.visited = set(), []
        self.runtime_ms, self.result = 0.0, None

    @property
    def frontier(self):
        return set(self.g) - self.closed

    def _finish(self, found):
        path = []
        if found:
            p = self.goal
            while True:
                path.append(p)
                if p == self.start:
                    break
                p = self.parent[p]
            path.reverse()
        self.result = dict(path=path, visited=list(self.visited),
                           cost=self.g[self.goal] if found else math.inf,
                           nodes_expanded=len(self.visited), found=found,
                           runtime_ms=self.runtime_ms, visited_order=list(self.visited),
                           runtime=self.runtime_ms/1000)

    def step(self):
        if self.result is not None:
            return None
        before = perf_counter()
        expanded = None
        while self.heap:
            _, g, _, p = heapq.heappop(self.heap)
            if p in self.closed or g != self.g[p]:
                continue  # Skip obsolete priority-queue entries.
            self.closed.add(p)
            self.visited.append(p)
            expanded = p
            if p == self.goal:
                self._finish(True)
                break
            for q, cost in self.grid.neighbors(p, self.movement):
                candidate = g + cost
                if candidate < self.g.get(q, math.inf):
                    self.g[q], self.parent[q] = candidate, p
                    heapq.heappush(self.heap, (candidate + self.h(q, self.goal),
                                             candidate, next(self.counter), q))
            break
        if not self.heap and self.result is None:
            self._finish(False)
        self.runtime_ms += (perf_counter() - before) * 1000
        if self.result is not None:
            self.result['runtime_ms'] = self.runtime_ms
            self.result['runtime'] = self.runtime_ms/1000
        return expanded

    def run(self):
        while self.result is None:
            self.step()
        return self.result


def dijkstra(grid, start, goal, movement=4):
    return SearchSession(grid, start, goal, 'dijkstra', movement=movement).run()


def astar(grid, start, goal, heuristic='manhattan', movement=4):
    return SearchSession(grid, start, goal, 'astar', heuristic, movement).run()
