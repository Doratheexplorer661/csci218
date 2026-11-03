"""Seeded random-grid generation for the CSCI218 experiments (Experiments and Analysis).

Design rules
------------
* Every random choice comes from a private ``random.Random(seed)``; the global
  ``random`` state is never touched, so the same seed always gives the same grid.
* Start and goal are chosen first and are never blocked or given terrain weights.
* Whether a grid is solvable is decided by a plain breadth-first reachability
  check over ``Grid.neighbors`` - NOT by Dijkstra or A*. Otherwise the
  benchmark would be using the algorithms under test to decide which grids they
  are allowed to see.
"""
import hashlib
from collections import deque
from dataclasses import dataclass
from random import Random

from .grid import Grid

ENDPOINT_MODES = ('corners', 'random')


@dataclass(frozen=True)
class GridConfig:
    """Parameters of one random-grid condition.

    density           fraction of non-endpoint cells that become walls (0.0-1.0)
    movement          4 or 8; used for the solvability check only
    terrain_fraction  fraction of free cells given a weight > 1 (0.0 = unit cost)
    terrain_weights   weights drawn (uniformly) for those cells; each must be >= 1
    endpoints         'corners' = (0,0) -> (rows-1, cols-1); 'random' = two random cells
    """
    rows: int
    cols: int
    density: float = 0.2
    movement: int = 4
    terrain_fraction: float = 0.0
    terrain_weights: tuple = (2.0, 3.0, 5.0)
    endpoints: str = 'corners'

    def __post_init__(self):
        if (type(self.rows) is not int or type(self.cols) is not int
                or self.rows <= 0 or self.cols <= 0):
            raise ValueError('rows and cols must be positive integers')
        if not 0.0 <= self.density <= 1.0:
            raise ValueError('density must be between 0 and 1')
        if self.movement not in (4, 8):
            raise ValueError('movement must be 4 or 8')
        if not 0.0 <= self.terrain_fraction <= 1.0:
            raise ValueError('terrain_fraction must be between 0 and 1')
        if self.terrain_fraction > 0 and (
                not self.terrain_weights or any(w < 1 for w in self.terrain_weights)):
            raise ValueError('terrain_weights must be non-empty and all >= 1')
        if self.endpoints not in ENDPOINT_MODES:
            raise ValueError(f'endpoints must be one of {ENDPOINT_MODES}')


@dataclass
class Instance:
    """One generated problem: grid + start + goal, plus how it was made."""
    grid: Grid
    start: tuple
    goal: tuple
    seed: int
    config: GridConfig
    solvable: bool
    attempts: int = 1

    @property
    def fingerprint(self):
        return grid_fingerprint(self.grid, self.start, self.goal)


def derive_seed(base_seed, *parts):
    """Stable 63-bit seed from a base seed and condition labels.

    Uses SHA-256 (not Python's salted ``hash``) so it is identical on every run
    and machine. Different conditions therefore get independent random streams
    instead of sharing one, which would make e.g. a 30% grid a superset of a
    20% grid built from the same seed.
    """
    text = ':'.join(str(p) for p in (base_seed,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 1


def grid_fingerprint(grid, start, goal):
    """Hash of everything a search can see: size, walls, weights, start, goal."""
    payload = repr((grid.rows, grid.cols, sorted(grid.walls),
                    sorted(grid.weights.items()), tuple(start), tuple(goal)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def is_solvable(grid, start, goal, movement=4):
    """Breadth-first reachability using the shared Grid movement rules.

    Independent of Dijkstra/A*: it only asks "is the goal reachable?", ignoring
    costs. Uses Grid.neighbors so 8-way corner-cutting rules match the searches.
    """
    if not grid.passable(start) or not grid.passable(goal):
        return False
    seen, queue = {start}, deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for nxt, _ in grid.neighbors(cell, movement):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _pick_endpoints(config, rng):
    if config.endpoints == 'corners':
        return (0, 0), (config.rows - 1, config.cols - 1)
    cells = config.rows * config.cols
    if cells < 2:
        return (0, 0), (0, 0)
    a, b = rng.sample(range(cells), 2)
    return divmod(a, config.cols), divmod(b, config.cols)


def _build(config, rng):
    start, goal = _pick_endpoints(config, rng)
    grid = Grid(config.rows, config.cols)
    for row in range(config.rows):
        for col in range(config.cols):
            cell = (row, col)
            if cell in (start, goal):
                continue
            # Fixed draw order (always two draws per cell) keeps seeds stable.
            wall_draw, terrain_draw = rng.random(), rng.random()
            if wall_draw < config.density:
                grid.set_wall(cell)
            elif terrain_draw < config.terrain_fraction:
                grid.weights[cell] = float(rng.choice(config.terrain_weights))
    return grid, start, goal


def generate_instance(config, seed, solvable_only=False, max_attempts=200):
    """Build the grid for ``seed``.

    solvable_only=False  keep whatever is generated; ``instance.solvable``
                         records whether a path exists (unsolvable cases are
                         therefore recorded, not silently dropped).
    solvable_only=True   redraw from the SAME seeded stream until a solvable grid
                         appears (deterministic), up to ``max_attempts``; raises
                         RuntimeError if none is found (e.g. density 1.0).
    """
    rng = Random(seed)
    for attempt in range(1, max_attempts + 1):
        grid, start, goal = _build(config, rng)
        solvable = is_solvable(grid, start, goal, config.movement)
        if solvable or not solvable_only:
            return Instance(grid, start, goal, seed, config, solvable, attempt)
    raise RuntimeError(f'No solvable grid found in {max_attempts} attempts '
                       f'(density={config.density}); lower the density.')


def generate_instances(config, base_seed, trials, solvable_only=False):
    """``trials`` instances with seeds derived from ``base_seed`` and the config."""
    for trial in range(trials):
        seed = derive_seed(base_seed, config.rows, config.cols, config.density,
                           config.terrain_fraction, config.endpoints, trial)
        yield generate_instance(config, seed, solvable_only)