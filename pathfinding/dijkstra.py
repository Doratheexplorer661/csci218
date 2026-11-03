"""Team Dijkstra algorithm adapted to an incremental session contract.

Retains coordinate tie-breaking and lazy deletion from the supplied code.
"""
import heapq
import math
from time import perf_counter
from .grid import Grid


class DijkstraSession:
    def __init__(self, grid, start, goal, movement=4, strict=True):
        if movement not in (4, 8):
            raise ValueError('movement must be 4 or 8')
        self.grid = grid.clone() if isinstance(grid, Grid) else Grid.from_matrix(grid)
        self.grid.__post_init__()
        self.start, self.goal, self.movement = tuple(start), tuple(goal), movement
        self.g, self.parent = {self.start: 0.0}, {}
        self.closed, self.visited = set(), []
        self.heap = [(0.0, self.start)]
        self.runtime_ms, self.result = 0.0, None
        if not self.grid.passable(self.start) or not self.grid.passable(self.goal):
            if strict:
                raise ValueError('Start and goal must be in bounds and unblocked')
            self._finish(False)

    @property
    def frontier(self):
        return set(self.g) - self.closed

    def _finish(self, found):
        path = []
        if found:
            path = [self.goal]
            while path[-1] in self.parent:
                path.append(self.parent[path[-1]])
            path.reverse()
        self.result = dict(found=found, path=path,
                           cost=self.g[self.goal] if found else math.inf,
                           nodes_expanded=len(self.visited), visited=list(self.visited),
                           visited_order=list(self.visited), runtime_ms=self.runtime_ms,
                           runtime=self.runtime_ms / 1000)

    def step(self):
        if self.result is not None:
            return None
        began = perf_counter()
        expanded = None
        while self.heap:
            distance, cell = heapq.heappop(self.heap)
            if cell in self.closed:
                continue
            self.closed.add(cell)
            self.visited.append(cell)
            expanded = cell
            if cell == self.goal:
                self._finish(True)
                break
            for neighbor, cost in self.grid.neighbors(cell, self.movement):
                if neighbor in self.closed:
                    continue
                candidate = distance + cost
                if candidate < self.g.get(neighbor, math.inf):
                    self.g[neighbor], self.parent[neighbor] = candidate, cell
                    heapq.heappush(self.heap, (candidate, neighbor))
            break
        if not self.heap and self.result is None:
            self._finish(False)
        self.runtime_ms += (perf_counter() - began) * 1000
        if self.result is not None:
            self.result.update(runtime_ms=self.runtime_ms, runtime=self.runtime_ms/1000)
        return expanded

    def run(self):
        while self.result is None:
            self.step()
        return self.result


def dijkstra(grid, start, goal, movement=4):
    # Preserve the supplied standalone API's no-path response for invalid endpoints.
    return DijkstraSession(grid, start, goal, movement, strict=False).run()
