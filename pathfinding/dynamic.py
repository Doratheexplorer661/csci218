"""Reactive replanning, not D* Lite: solve again from the actual robot cell."""
from .search import SearchSession


class Navigator:
    def __init__(self, grid, start, goal, algorithm='astar', heuristic='manhattan', movement=4):
        self.grid, self.position, self.goal = grid, start, goal
        self.algorithm, self.heuristic, self.movement = algorithm, heuristic, movement
        self.replans, self.travel_cost, self.history = 0, 0.0, [start]
        self.result = None
        self.plan()

    def plan(self):
        self.result = SearchSession(self.grid, self.position, self.goal, self.algorithm,
                                    self.heuristic, self.movement).run()
        self.remaining = self.result['path'][1:]
        self.replans += 1
        return self.result

    def edit(self, cell, wall=None, weight=None):
        if not self.grid.inside(cell):
            raise ValueError('Cell outside grid')
        if wall and cell in (self.position, self.goal):
            raise ValueError('Cannot block robot or goal')
        if weight is not None:
            import math
            if not math.isfinite(weight) or weight < 1:
                raise ValueError('Weight must be finite and >= 1')
        if wall is True:
            self.grid.walls.add(cell)
        elif wall is False:
            self.grid.walls.discard(cell)
        if weight is not None:
            self.grid.weights[cell] = weight
        # Replan even for off-path removals/cheaper terrain: a better route may exist.
        return self.plan()

    def tick(self):
        if self.position == self.goal or not self.remaining:
            return False
        q = self.remaining[0]
        costs = dict(self.grid.neighbors(self.position, self.movement))
        if q not in costs:
            self.plan()  # Defensive check includes newly blocked diagonal side cells.
            return False
        self.position = self.remaining.pop(0)
        self.history.append(self.position)
        self.travel_cost += costs[q]
        return True
