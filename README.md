# CSCI218 Group Project

This folder integrates the supplied team code into one runnable Python package. The original source folder is unchanged. The interface uses the team's **Tkinter** implementation; this version does not require pygame.

## Install and run on Windows

Install Python 3.10+ with Tcl/Tk support (the standard Windows Python installer includes it). Open PowerShell in this folder, where `run_ui.py` is located:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_ui.py
```

If `python` is not available, use `py` for the first command. Reopen the terminal after installing Python. Virtual environment activation is not required. `python main.py` is an alternative entry point.

The interface and search need only the Python standard library, including Tkinter. Matplotlib is required for benchmark charts and the chart tests. To check Tkinter installation, run `python -m tkinter`.

## Use the interface

1. Select Wall, Erase, Start, Goal or Terrain x3, then click a cell. Erase removes a wall and restores terrain cost 1. Clear walls removes obstacles but keeps terrain.
2. Select A* or Dijkstra, four-way or eight-way movement, and an A* heuristic. Eight-way Manhattan automatically changes to diagonal, since unscaled Manhattan would overestimate diagonal costs.
3. Click Run animation, Pause, Next step or Reset animation. The initial search computes a result, then replays the recorded expansion order and path. It is a visualization of that order, not an asynchronous search. Large grids may block the interface briefly during computation.
4. Click Robot go / stop to follow a planned route. Yellow R marks the robot. Click Block route or press B while the window has focus to add a wall on its remaining route.
5. Edit walls or terrain while the robot is moving: it replans from its actual position. If no route exists it waits; removing a blockage can restore a route. Pause also stops robot movement; click Robot go / stop to resume it.
6. Enter an integer Seed and click Generate map for a reproducible random map. No corridor is forced, so a generated map may be disconnected.

Blue cells are explored nodes, green cells form the route, and tan cells have terrain multiplier 3. Search colours can cover the terrain colour while its cost remains active. This interface uses click editing, not drag painting. Apart from B, controls are buttons rather than the previous pygame project's shortcuts.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_ui.py --smoke
```

The smoke option opens the real window for half a second and closes it. The tests include the 76 supplied tests and five new contract/UI tests. Tkinter UI tests create and withdraw real windows; they are skipped on machines without an available display.

## Benchmarks

```powershell
.\.venv\Scripts\python.exe run_benchmarks.py --check
.\.venv\Scripts\python.exe run_benchmarks.py --sizes 20x20 40x40 --densities 0.1 0.2 --trials 30 --runtime-repeats 3 --allow-unsolvable --output-dir results/local
.\.venv\Scripts\python.exe run_benchmarks.py --sizes 20x20 --densities 0.2 --trials 30 --terrain-fraction 0.2 --allow-unsolvable --output-dir results/local_weighted
.\.venv\Scripts\python.exe -m pathfinding.analysis --raw results/local/raw_results.csv --output-dir results/replotted
```

Each batch produces `raw_results.csv`, `summary_stats.csv` and six PNG charts (three metrics for each movement mode). Four-way runs compare Dijkstra with all three A* heuristics; eight-way runs exclude Manhattan. Each algorithm receives the same instance for a trial. Both movement modes also use the same instance.

The supplied experiment design defaults to redrawing until maps are solvable. Use **--allow-unsolvable**, as in the included batches, to retain disconnected maps. All metric means and sample standard deviations use successful trials only; success rate includes all trials. A failed path cost is blank. A standard deviation based on fewer than two successes is blank in CSV. Missing means are not displayed as zero-valued bars.

Runtime is the median of three accumulated search-step times, excluding snapshot creation, UI delays, map generation and plotting. It is not end-to-end latency. Search order and grid hashes are deterministic; timings vary by machine. Dijkstra uses the member's coordinate tie-break, while A* uses its own tie-break. Runtime comparisons therefore describe these implementations, not just abstract algorithms. The benchmark uses a fixed algorithm execution order; warm-up and randomized timing order would strengthen future experiments.

## Structure and shared rules

- `pathfinding/grid.py`: map, movement rules, compatibility methods, seeded-grid helper.
- `pathfinding/search.py`: A* implementation and common session entry point.
- `pathfinding/dijkstra.py`: member's Dijkstra algorithm adapted to incremental steps.
- `pathfinding/heuristics.py`: Manhattan, Euclidean and Octile distances.
- `pathfinding/dynamic.py`: current-position navigation and reactive replanning.
- `pathfinding/visualization.py`, `playback.py`, `search_adapter.py`: Tkinter interface and search playback.
- `pathfinding/experiments.py`, `benchmark.py`, `analysis.py`: seeded generation, measurements, statistics and charts.
- `tests/`: original tests plus integration coverage.
- `results/`: actual benchmark outputs from this integration, separate from the earlier project.

Coordinates are `(row, column)`. Cardinal moves cost 1, diagonals cost sqrt(2), multiplied by the destination terrain weight. Weights must be finite and at least 1. Both side cells must be free for a diagonal move; corner cutting is forbidden. Reactive replanning calls the selected search again and is not D* Lite.

Result dictionaries contain `found`, `path`, `cost`, `nodes_expanded`, `visited`, `visited_order`, `runtime_ms` and `runtime`. The visited names are aliases, and runtime is in seconds while runtime_ms is in milliseconds. Paths include both endpoints. No route means an empty path and infinite cost. The common SearchSession API rejects blocked/outside endpoints; the member's standalone `pathfinding.dijkstra.dijkstra()` retains its original no-path response for those inputs.

For common strict behavior, use:

```python
from pathfinding import Grid, astar, dijkstra
grid = Grid(10, 10)
result = astar(grid, (0, 0), (9, 9), heuristic='manhattan', movement=4)
baseline = dijkstra(grid, (0, 0), (9, 9), movement=4)
```

The `pathfinding` folder must remain together. Run entry points from the project root instead of running package files directly. All changes in this version are integration work; contribution ownership should be based on the team's actual development records.
