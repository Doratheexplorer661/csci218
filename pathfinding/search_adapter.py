class TracedGrid:

    def __init__(self, grid):
        self.grid = grid
        self.expansion_calls = []

    def __getattr__(self, attribute):
        return getattr(self.grid, attribute)

    def neighbors(self, cell, movement=4):
        self.expansion_calls.append(cell)
        return self.grid.neighbors(cell, movement)


def get_search_result(grid, start, goal, algorithm='astar',
                      heuristic='manhattan', movement=4):
    try:
        from .search import SearchSession
    except ModuleNotFoundError as error:
        if error.name == 'pathfinding.search':
            raise RuntimeError(
                "Search module not available."
            ) from error
        raise

    result = SearchSession(grid, start, goal, algorithm, heuristic, movement).run()
    if not isinstance(result, dict) or 'found' not in result or 'path' not in result:
        raise ValueError("SearchSession.run() must return a dict with 'found' and 'path'")

    raw_order = result.get('visited_order')
    if raw_order is None:
        raw_order = result.get('visited', [])
    explored = []
    seen = set()
    for raw in raw_order:
        cell = tuple(raw)
        if grid.inside(cell) and cell not in seen:
            explored.append(cell)
            seen.add(cell)

    path = [tuple(cell) for cell in result['path']]
    if result['found']:
        if not path or path[0] != start or path[-1] != goal:
            raise ValueError('The returned path must start at S and end at G')
        if any(b not in dict(grid.neighbors(a, movement)) for a, b in zip(path, path[1:])):
            raise ValueError('The returned path contains an invalid move')
    elif path:
        raise ValueError('An unsuccessful search should return an empty path')
    return dict(result, path=path), explored
