import math


class Grid:
    def __init__(self, rows, cols, walls=(), weights=None):
        if (type(rows) is not int or type(cols) is not int
                or rows <= 0 or cols <= 0):
            raise ValueError("Grid dimensions must be positive integers")
        self.rows = rows
        self.cols = cols
        self.walls = set(walls)

        self.weights = dict(weights or {})
        self.__post_init__()

    def __post_init__(self):
        if any(not self.inside(p) for p in self.walls | self.weights.keys()):
            raise ValueError('Cell outside grid')
        if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 1
               for w in self.weights.values()):
            raise ValueError('Terrain weights must be finite and >= 1')

    def clone(self):
        return Grid(self.rows, self.cols, self.walls, self.weights)

    def __eq__(self, other):
        return (isinstance(other, Grid) and self.rows == other.rows and self.cols == other.cols
                and self.walls == other.walls and self.weights == other.weights)

    def free(self, cell):
        return self.passable(cell)

    @classmethod
    def from_matrix(cls, matrix):
        if not matrix or not matrix[0] or any(len(r) != len(matrix[0]) for r in matrix):
            raise ValueError('Matrix must be nonempty and rectangular')
        if any(v not in (0, 1) for row in matrix for v in row):
            raise ValueError('Matrix must use 0 for free and 1 for wall')
        return cls(len(matrix), len(matrix[0]),
                   {(r, c) for r, row in enumerate(matrix) for c, v in enumerate(row) if v})

    def inside(self, cell):
        if (not isinstance(cell, tuple) or len(cell) != 2
                or any(type(x) is not int for x in cell)):
            return False
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def passable(self, cell):
        return self.inside(cell) and cell not in self.walls

    def set_wall(self, cell, blocked=True):
        if not self.inside(cell):
            raise ValueError("Cell is outside the grid")
        if blocked:
            self.walls.add(cell)
            self.weights.pop(cell, None)
        else:
            self.walls.discard(cell)

    def neighbors(self, cell, movement=4):
        if movement not in (4, 8):
            raise ValueError("movement must be 4 or 8")
        if not self.passable(cell):
            return []

        row, col = cell
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if movement == 8:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        available = []
        for dr, dc in directions:
            next_cell = (row + dr, col + dc)
            if not self.passable(next_cell):
                continue
            if dr != 0 and dc != 0:

                if not (self.passable((row + dr, col))
                        and self.passable((row, col + dc))):
                    continue
            distance = math.sqrt(2) if dr and dc else 1.0
            cost = self.weights.get(next_cell, 1.0)
            if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 1:
                raise ValueError("Terrain weights must be finite and >= 1")
            available.append((next_cell, distance * cost))
        return available


def random_grid(rows=20, cols=20, density=.2, seed=42, weighted=False):
    from .experiments import GridConfig, generate_instance
    return generate_instance(GridConfig(rows, cols, density,
                             terrain_fraction=.2 if weighted else 0), seed).grid
