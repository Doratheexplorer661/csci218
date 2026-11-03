"""Benchmark runner: Dijkstra vs A* (Manhattan / Euclidean / diagonal).

Fairness controls
-----------------
1. One generated instance (grid + start + goal) is built per (size, density,
   trial) and given to EVERY algorithm and movement mode - never regenerated.
2. All searches go through the ``SearchSession`` interface, dispatching to the
   team's Dijkstra session or A* engine (``nodes_expanded`` = cells popped and closed;
   ``runtime_ms`` = time spent inside ``step()`` only).
3. Inadmissible combinations (Manhattan A* with 8-way moves) are rejected up
   front instead of being run and reported as if they were comparable.
4. Every A* run is checked against Dijkstra's cost on the same instance
   (``optimal`` column) and every result is checked against an independent BFS
   reachability test (``solvable`` column).

Usage
-----
    python -m pathfinding.benchmark --trials 30 --seed 218 \
        --sizes 20x20 50x50 --densities 0.1 0.2 0.3 --movements 4 8
"""
import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass, field

from .experiments import (GridConfig, ENDPOINT_MODES, generate_instances,
                          is_solvable)
from .grid import Grid
from .heuristics import HEURISTICS
from .search import SearchSession

# (algorithm, heuristic). Dijkstra has no heuristic (it is A* with h = 0).
ALGORITHM_PLAN = (
    ('dijkstra', None),
    ('astar', 'manhattan'),
    ('astar', 'euclidean'),
    ('astar', 'diagonal'),
)

CSV_FIELDS = ['seed', 'trial', 'rows', 'cols', 'density', 'terrain_fraction',
              'n_walls', 'movement', 'algorithm', 'heuristic', 'solvable',
              'found', 'cost', 'nodes_expanded', 'runtime_ms', 'optimal',
              'grid_hash']

COST_TOLERANCE = 1e-9  # float tolerance when comparing sums of sqrt(2) steps


@dataclass
class ExperimentConfig:
    sizes: tuple = ((20, 20),)
    densities: tuple = (0.1, 0.2, 0.3)
    movements: tuple = (4, 8)
    trials: int = 30
    base_seed: int = 218
    terrain_fraction: float = 0.0
    terrain_weights: tuple = (2.0, 3.0, 5.0)
    solvable_only: bool = True
    runtime_repeats: int = 3
    endpoints: str = 'corners'

    def __post_init__(self):
        if self.trials < 1 or self.runtime_repeats < 1:
            raise ValueError('trials and runtime_repeats must be >= 1')
        if not self.sizes or not self.densities or not self.movements:
            raise ValueError('sizes, densities and movements must be non-empty')
        if any(m not in (4, 8) for m in self.movements):
            raise ValueError('movements must be 4 and/or 8')
        if self.endpoints not in ENDPOINT_MODES:
            raise ValueError(f'endpoints must be one of {ENDPOINT_MODES}')


# --------------------------------------------------------------------------
# Algorithm selection
# --------------------------------------------------------------------------
def algorithm_label(algorithm, heuristic):
    return 'dijkstra' if algorithm == 'dijkstra' else f'astar-{heuristic}'


def check_combination(algorithm, heuristic, movement):
    """Raise ValueError for unknown or inadmissible algorithm/heuristic/movement."""
    if algorithm not in ('dijkstra', 'astar'):
        raise ValueError(f'Unknown algorithm: {algorithm}')
    if movement not in (4, 8):
        raise ValueError('movement must be 4 or 8')
    if algorithm == 'astar':
        if heuristic not in HEURISTICS:
            raise ValueError(f'Unknown heuristic: {heuristic}')
        if movement == 8 and heuristic == 'manhattan':
            raise ValueError('Manhattan is not admissible for 8-way movement '
                             '(it can overestimate the sqrt(2) diagonal cost)')


def algorithms_for(movement):
    """The (algorithm, heuristic) pairs that may legally run with ``movement``."""
    plan = []
    for algorithm, heuristic in ALGORITHM_PLAN:
        try:
            check_combination(algorithm, heuristic, movement)
        except ValueError:
            continue  # skipped by design; reported in the methodology
        plan.append((algorithm, heuristic))
    return plan


def check_integration():
    """Fail early, with a readable message, if Grid lacks what SearchSession needs."""
    missing = [name for name in ('clone', 'free', '__post_init__')
               if not hasattr(Grid, name)]
    if missing:
        raise RuntimeError(
            'pathfinding/grid.py is missing ' + ', '.join(missing) + ', which '
            'pathfinding/search.py (SearchSession) calls. Ask the Grid owner to '
            'add them (see the integration notes) before running experiments.')


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def run_single(instance, algorithm, heuristic, movement, runtime_repeats=1):
    """Run one algorithm on one instance via SearchSession and return a metric row.

    Deterministic metrics (cost, nodes expanded) come from the first run and
    are checked to be identical on every repeat. Runtime is the median of
    ``runtime_repeats`` runs, which damps OS scheduling noise.
    Unsuccessful runs have cost=None (never infinity) so they can never leak
    into a cost average.
    """
    check_combination(algorithm, heuristic, movement)
    runtimes, first = [], None
    for _ in range(runtime_repeats):
        result = SearchSession(instance.grid, instance.start, instance.goal,
                               algorithm, heuristic or 'manhattan', movement).run()
        runtimes.append(result['runtime_ms'])
        if first is None:
            first = result
        elif (result['found'] != first['found']
              or result['nodes_expanded'] != first['nodes_expanded']
              or not (result['cost'] == first['cost'] or math.isclose(result['cost'], first['cost'], abs_tol=COST_TOLERANCE))):
            raise RuntimeError('Search is not deterministic across repeats')
    found = bool(first['found'])
    config = instance.config
    return {
        'seed': instance.seed,
        'rows': config.rows,
        'cols': config.cols,
        'density': config.density,
        'terrain_fraction': config.terrain_fraction,
        'n_walls': len(instance.grid.walls),
        'movement': movement,
        'algorithm': algorithm,
        'heuristic': heuristic or 'none',
        'solvable': is_solvable(instance.grid, instance.start, instance.goal, movement),
        'found': found,
        'cost': first['cost'] if found else None,
        'nodes_expanded': first['nodes_expanded'],
        'runtime_ms': statistics.median(runtimes),
        'optimal': None,  # filled in by run_benchmark once Dijkstra's cost is known
        'grid_hash': instance.fingerprint,
    }


# --------------------------------------------------------------------------
# Whole experiment
# --------------------------------------------------------------------------
def run_benchmark(config, progress=None):
    """Run the full experiment and return a list of raw result rows."""
    check_integration()
    rows = []
    for (n_rows, n_cols) in config.sizes:
        for density in config.densities:
            # Solvability is guaranteed under 4-way movement (the most
            # restrictive), which implies solvable under 8-way as well.
            grid_config = GridConfig(n_rows, n_cols, density, movement=4,
                                     terrain_fraction=config.terrain_fraction,
                                     terrain_weights=config.terrain_weights,
                                     endpoints=config.endpoints)
            instances = generate_instances(grid_config, config.base_seed,
                                           config.trials, config.solvable_only)
            for trial, instance in enumerate(instances):
                for movement in config.movements:
                    reference = None
                    for algorithm, heuristic in algorithms_for(movement):
                        row = run_single(instance, algorithm, heuristic, movement,
                                         config.runtime_repeats)
                        row['trial'] = trial
                        if algorithm == 'dijkstra':
                            reference = row  # Dijkstra is the optimality reference
                        row['optimal'] = _matches_reference(row, reference)
                        rows.append(row)
            if progress:
                progress(f'{n_rows}x{n_cols} density={density}: '
                         f'{config.trials} trials done')
    return rows


def _matches_reference(row, reference):
    if reference is None:
        return None
    if row['found'] != reference['found']:
        return False
    if not row['found']:
        return True  # both report "no path"
    return abs(row['cost'] - reference['cost']) <= COST_TOLERANCE


# --------------------------------------------------------------------------
# CSV + CLI
# --------------------------------------------------------------------------
def write_raw_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if row[k] is None else row[k]) for k in CSV_FIELDS})


def parse_size(text):
    try:
        rows, cols = text.lower().split('x')
        return int(rows), int(cols)
    except ValueError:
        raise argparse.ArgumentTypeError(f'size must look like 20x30, got {text!r}')


def build_parser():
    p = argparse.ArgumentParser(description='Benchmark Dijkstra vs A* on seeded random grids.')
    p.add_argument('--sizes', nargs='+', type=parse_size, default=[(20, 20)],
                   help='grid sizes, e.g. 20x20 50x50')
    p.add_argument('--densities', nargs='+', type=float, default=[0.1, 0.2, 0.3])
    p.add_argument('--movements', nargs='+', type=int, choices=(4, 8), default=[4, 8])
    p.add_argument('--trials', type=int, default=30, help='random grids per condition')
    p.add_argument('--seed', type=int, default=218, help='base seed')
    p.add_argument('--terrain-fraction', type=float, default=0.0,
                   help='fraction of free cells with weight > 1')
    p.add_argument('--endpoints', choices=ENDPOINT_MODES, default='corners')
    p.add_argument('--allow-unsolvable', action='store_true',
                   help='keep unsolvable grids and record them (default: redraw until solvable)')
    p.add_argument('--runtime-repeats', type=int, default=3,
                   help='runs per instance; the median runtime is stored')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--check', action='store_true', help='only check integration and exit')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    check_integration()
    if args.check:
        print('Integration check passed: Grid provides clone/free/__post_init__.')
        return 0
    from . import analysis  # imported here so matplotlib is only needed when plotting
    config = ExperimentConfig(
        sizes=tuple(args.sizes), densities=tuple(args.densities),
        movements=tuple(args.movements), trials=args.trials, base_seed=args.seed,
        terrain_fraction=args.terrain_fraction, solvable_only=not args.allow_unsolvable,
        runtime_repeats=args.runtime_repeats, endpoints=args.endpoints)
    rows = run_benchmark(config, progress=lambda m: print(m, file=sys.stderr))
    raw_path = os.path.join(args.output_dir, 'raw_results.csv')
    write_raw_csv(rows, raw_path)
    analysis.run_analysis(rows, args.output_dir)
    print(f'Wrote {len(rows)} raw rows to {raw_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
