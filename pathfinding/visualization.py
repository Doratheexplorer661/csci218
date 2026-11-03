import tkinter as tk
from tkinter import ttk, messagebox

from .grid import Grid
from .playback import SearchPlayback
from .search_adapter import get_search_result
from .dynamic import Navigator
from .experiments import GridConfig, generate_instance

CELL_SIZE = 34
COLORS = {
    'free': '#f0f4f8',
    'wall': '#465467',
    'explored': '#a9d7f2',
    'route': '#80cfa9',
    'start': '#21825d',
    'goal': '#e49536',
    'robot': '#f4d35e',
    'terrain': '#d9b47b',
}


class EnvironmentUI:
    def __init__(self, root, rows=10, cols=15):
        self.root = root
        root.title('CSCI218 - Environment and Visualisation')
        self.grid = Grid(rows, cols)
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.edit_mode = tk.StringVar(value='wall')
        self.algorithm = tk.StringVar(value='astar')
        self.heuristic = tk.StringVar(value='manhattan')
        self.movement = tk.StringVar(value='4')
        self.status = tk.StringVar(value='Click a square to edit the environment.')
        self.playback = None
        self.result = None
        self._timer = None
        self._playing = False
        self.navigator = None
        self._walk_timer = None
        self._walking = False
        self.seed = tk.StringVar(value='42')
        self._build()
        self.draw()
        self.root.protocol('WM_DELETE_WINDOW', self._close)
        self.root.bind('<Key-b>', lambda event: self.block_route())
        self.root.bind('<Key-B>', lambda event: self.block_route())
        for variable in (self.algorithm, self.heuristic, self.movement):
            variable.trace_add('write', self._settings_changed)

    def _build(self):
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill='x')
        ttk.Label(panel, text='Edit grid:', font=('Arial', 11, 'bold')).pack(side='left', padx=(0, 8))
        for mode, label in [('wall', 'Wall'), ('erase', 'Erase'),
                            ('start', 'Start'), ('goal', 'Goal'), ('terrain', 'Terrain x3')]:
            ttk.Radiobutton(panel, text=label, variable=self.edit_mode,
                            value=mode).pack(side='left', padx=5)
        ttk.Button(panel, text='Clear walls', command=self.clear_walls).pack(side='right')

        settings = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        settings.pack(fill='x')
        ttk.Label(settings, text='Algorithm').pack(side='left')
        ttk.Combobox(settings, textvariable=self.algorithm, state='readonly',
                     values=('astar', 'dijkstra'), width=9).pack(side='left', padx=5)
        ttk.Label(settings, text='Moves').pack(side='left', padx=(10, 0))
        ttk.Combobox(settings, textvariable=self.movement, state='readonly',
                     values=('4', '8'), width=3).pack(side='left', padx=5)
        ttk.Label(settings, text='A* heuristic').pack(side='left', padx=(10, 0))
        ttk.Combobox(settings, textvariable=self.heuristic, state='readonly',
                     values=('manhattan', 'euclidean', 'diagonal'),
                     width=11).pack(side='left', padx=5)

        self.canvas = tk.Canvas(self.root, width=self.grid.cols * CELL_SIZE,
                                height=self.grid.rows * CELL_SIZE,
                                highlightthickness=0)
        self.canvas.pack(padx=12, pady=5)
        self.canvas.bind('<Button-1>', self.on_click)

        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill='x')
        ttk.Button(controls, text='Run animation', command=self.run).pack(side='left', padx=4)
        ttk.Button(controls, text='Pause', command=self.pause).pack(side='left', padx=4)
        ttk.Button(controls, text='Next step', command=self.step).pack(side='left', padx=4)
        ttk.Button(controls, text='Reset animation', command=self.reset_animation).pack(side='left', padx=4)
        ttk.Label(controls, text='Blue = explored, green = final route').pack(side='right')
        dynamic_controls = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        dynamic_controls.pack(fill='x')
        ttk.Button(dynamic_controls, text='Robot go / stop', command=self.toggle_robot).pack(side='left', padx=4)
        ttk.Button(dynamic_controls, text='Block route (B)', command=self.block_route).pack(side='left', padx=4)
        ttk.Label(dynamic_controls, text='Seed').pack(side='left', padx=(12, 3))
        ttk.Entry(dynamic_controls, textvariable=self.seed, width=8).pack(side='left')
        ttk.Button(dynamic_controls, text='Generate map', command=self.generate_map).pack(side='left', padx=4)
        ttk.Label(self.root, textvariable=self.status, padding=(15, 0, 15, 15),
                  wraplength=self.grid.cols * CELL_SIZE).pack(anchor='w')

    def on_click(self, event):
        cell = (event.y // CELL_SIZE, event.x // CELL_SIZE)
        if not self.grid.inside(cell):
            return
        mode = self.edit_mode.get()
        if self.navigator and mode in ('wall', 'erase', 'terrain'):
            if mode == 'wall' and cell in (self.navigator.position, self.goal):
                return
            self.navigator.edit(cell, wall=(mode == 'wall'), weight=3 if mode == 'terrain' else 1)
            self._show_navigation()
            return
        if mode == 'wall' and cell not in (self.start, self.goal):
            self.grid.set_wall(cell)
        elif mode == 'erase':
            self.grid.set_wall(cell, False)
        elif mode == 'start' and cell != self.goal:
            self.grid.set_wall(cell, False)
            self.start = cell
        elif mode == 'goal' and cell != self.start:
            self.grid.set_wall(cell, False)
            self.goal = cell
        elif mode == 'terrain':
            self.grid.set_wall(cell, False)
            self.grid.weights[cell] = 3
        else:
            return
        if mode == 'erase':
            self.grid.weights.pop(cell, None)
        if self.navigator and mode != 'start':
            self.start = self.navigator.position
        self.reset_animation()
        self.status.set(f'Map updated at {cell}.')

    def draw(self):
        self.canvas.delete('all')
        explored = self.playback.explored if self.playback else set()
        route = self.playback.route if self.playback else set()
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                cell = (row, col)
                if self.navigator and cell == self.navigator.position:
                    color = COLORS['robot']
                elif cell == self.start:
                    color = COLORS['start']
                elif cell == self.goal:
                    color = COLORS['goal']
                elif cell in self.grid.walls:
                    color = COLORS['wall']
                elif cell in route:
                    color = COLORS['route']
                elif cell in explored:
                    color = COLORS['explored']
                elif self.grid.weights.get(cell, 1) > 1:
                    color = COLORS['terrain']
                else:
                    color = COLORS['free']
                x, y = col * CELL_SIZE, row * CELL_SIZE
                self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE,
                                             fill=color, outline='white')
                if cell in (self.start, self.goal) or (self.navigator and cell == self.navigator.position):
                    label = 'R' if self.navigator and cell == self.navigator.position else 'S' if cell == self.start else 'G'
                    self.canvas.create_text(x + CELL_SIZE/2, y + CELL_SIZE/2,
                                            text=label, fill='white',
                                            font=('Arial', 12, 'bold'))

    def clear_walls(self):
        self.grid.walls.clear()
        if self.navigator:
            self.navigator.plan()
            self._show_navigation()
            return
        self.reset_animation()
        self.status.set('All walls removed.')

    def _cancel_timer(self):
        self._playing = False
        if self._timer is not None:
            self.root.after_cancel(self._timer)
            self._timer = None

    def reset_animation(self):
        self._cancel_timer()
        self._stop_robot()
        self.navigator = None
        self.playback = None
        self.result = None
        self.draw()

    def _prepare_animation(self):
        self._stop_robot()
        if self.navigator:
            self.start = self.navigator.position
            self.navigator = None
        try:
            self.result, order = get_search_result(
                self.grid, self.start, self.goal,
                self.algorithm.get(), self.heuristic.get(), int(self.movement.get()))
        except (RuntimeError, ValueError, ImportError) as error:
            self.status.set(str(error))
            messagebox.showinfo('Search code not ready', str(error))
            return False
        self.playback = SearchPlayback(order, self.result['path'])
        self.status.set(f'Search returned: {len(order)} explored steps, '
                        f'{len(self.result["path"])} path cells.')
        return True

    def _advance(self):
        if self.playback is None:
            return False
        event = self.playback.step()
        if event is None:
            self._complete()
            return False
        self.draw()
        if self.playback.finished:
            self._complete()
        return True

    def _complete(self):
        self._cancel_timer()
        if self.result is not None:
            self.status.set('Finished: ' + ('path found.' if self.result['found']
                                             else 'no path found.'))

    def step(self):
        if self.navigator:
            self.start = self.navigator.position
            self.reset_animation()
        self._cancel_timer()
        if self.playback is None and not self._prepare_animation():
            return
        self._advance()

    def run(self):
        if self.navigator:
            self.start = self.navigator.position
            self.reset_animation()
        if self.playback is None and not self._prepare_animation():
            return
        if not self.playback.finished:
            self._playing = True
            self._animate()

    def _animate(self):
        self._timer = None
        if not self._playing:
            return
        if self._advance() and self._playing:
            self._timer = self.root.after(70, self._animate)

    def pause(self):
        self._cancel_timer()
        self._stop_robot()
        self.status.set('Animation paused.')

    def _close(self):
        self._cancel_timer()
        self._stop_robot()
        self.root.destroy()

    def _settings_changed(self, *args):
        if self.movement.get() == '8' and self.heuristic.get() == 'manhattan':
            self.heuristic.set('diagonal')
        if self.navigator:
            self.start = self.navigator.position
        self.reset_animation()
        self.status.set('Settings changed. Eight-way A* uses Euclidean or diagonal.')

    def generate_map(self):
        try:
            instance = generate_instance(GridConfig(self.grid.rows, self.grid.cols, .2), int(self.seed.get()))
        except ValueError:
            self.status.set('Seed must be an integer.')
            return
        self.grid, self.start, self.goal = instance.grid, instance.start, instance.goal
        self.reset_animation()
        self.status.set('Seeded map generated; connectivity is not guaranteed.')

    def _stop_robot(self):
        self._walking = False
        if self._walk_timer is not None:
            self.root.after_cancel(self._walk_timer)
            self._walk_timer = None

    def _show_navigation(self):
        self.result = self.navigator.result
        self.playback = SearchPlayback([], [])
        self.playback.explored = set(self.result['visited'])
        self.playback.route = set(self.navigator.remaining)
        self.draw()
        outcome = 'goal reached' if self.navigator.position == self.goal else 'route ready' if self.result['found'] else 'no route; waiting for map edits'
        self.status.set(f'{outcome}; plans={self.navigator.replans}, travelled={self.navigator.travel_cost:.3f}, '
                        f'planned cost={self.result["cost"]:.3f}, nodes={self.result["nodes_expanded"]}')

    def toggle_robot(self):
        if self._walking:
            self._stop_robot()
            return
        self._cancel_timer()
        if self.navigator is None:
            self.navigator = Navigator(self.grid, self.start, self.goal, self.algorithm.get(),
                                       self.heuristic.get(), int(self.movement.get()))
        self._walking = True
        self._show_navigation()
        self._walk()

    def _walk(self):
        self._walk_timer = None
        if not self._walking:
            return
        self.navigator.tick()
        self._show_navigation()
        if self.navigator.position == self.goal:
            self._walking = False
        else:
            self._walk_timer = self.root.after(200, self._walk)

    def block_route(self):
        if not self.navigator:
            self.status.set('Start the robot first, then block its route.')
            return
        candidates = [p for p in self.navigator.remaining if p != self.goal]
        if candidates:
            self.navigator.edit(candidates[len(candidates)//2], wall=True)
            self._show_navigation()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Integrated CSCI218 pathfinding interface')
    parser.add_argument('--smoke', action='store_true', help='Open the window briefly, then exit')
    args = parser.parse_args()
    root = tk.Tk()
    ui = EnvironmentUI(root)
    if args.smoke:
        root.after(500, ui._close)
    root.mainloop()


if __name__ == '__main__':
    main()
