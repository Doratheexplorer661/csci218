class SearchPlayback:
    def __init__(self, explored_cells, final_path):
        self.events = ([('explore', tuple(cell)) for cell in explored_cells]
                       + [('route', tuple(cell)) for cell in final_path])
        self.index = 0
        self.explored = set()
        self.route = set()

    def step(self):
        if self.index >= len(self.events):
            return None
        kind, cell = self.events[self.index]
        self.index += 1
        if kind == 'explore':
            self.explored.add(cell)
        else:
            self.route.add(cell)
        return kind, cell

    @property
    def finished(self):
        return self.index >= len(self.events)
