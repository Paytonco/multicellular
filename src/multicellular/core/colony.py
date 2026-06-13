# core/colony.py


class Colony:
    """
    A collection of Cells living within an Environment.
    """

    def __init__(self, cells, environment):
        self.cells = list(cells)
        self.environment = environment
        existing_ids = [cell.id for cell in self.cells if cell.id is not None]
        self._next_id = max(existing_ids, default=-1) + 1

    @property
    def living_cells(self):
        return [cell for cell in self.cells if cell.alive]

    def step(self, dt):
        """Advance all cells by one timestep, then enforce bounds and divisions."""
        for cell in self.cells:
            cell.step(dt)
        self.enforce_bounds()
        self.handle_divisions()

    def enforce_bounds(self):
        """Kill any cell whose center of mass has left the environment bounds."""
        for cell in self.cells:
            if cell.alive and not self.environment.in_bounds(cell.position):
                cell.kill()

    def handle_divisions(self):
        """Replace any cell ready to divide with its two daughter cells."""
        new_cells = []
        for cell in self.cells:
            daughters = cell.divide() if cell.alive else None
            if daughters is None:
                new_cells.append(cell)
                continue
            for daughter in daughters:
                daughter.id = self._next_id
                self._next_id += 1
                new_cells.append(daughter)
        self.cells = new_cells
