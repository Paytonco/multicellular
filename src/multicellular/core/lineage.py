# core/lineage.py

from dataclasses import asdict, dataclass

import pandas as pd

# Terminal fates a lineage record can end in. "alive" is the only
# non-terminal one: its record has no `end_time` yet.
FATES = ("alive", "divided", "out_of_bounds", "survival_condition", "killed")


@dataclass
class LineageRecord:
    """
    One node of a lineage tree: a single cell, from its birth (as a founder
    or at its parent's division) to its division or death.

    Times are simulation times. `end_time` is None while the cell is alive;
    `size_at_division` is None unless the cell divided. Sizes are in the
    cell's size-control units: length (μm) for rods, volume (μm³) for
    spheres (see `Cell`).
    """

    id: int
    parent_id: int | None
    root_id: int
    generation: int
    birth_time: float
    end_time: float | None = None
    fate: str = "alive"
    daughter_ids: tuple = ()
    size_at_birth: float | None = None
    size_at_division: float | None = None
    shape: str = "rod"
    species: str = "default"


class Lineage:
    """
    Registry of every cell a Colony has ever held, as a forest of lineage
    trees (one tree per founder cell).

    Colony writes to it (founders at construction, divisions, deaths); users
    query it. `time` is the colony's latest simulation time, used to close
    off the branches of cells that are still alive (e.g. in `to_newick` and
    plots).
    """

    def __init__(self):
        self.records = {}
        self.time = 0.0

    # ------------------------------------------------------------------
    # Writers (called by Colony)
    # ------------------------------------------------------------------

    def _new_record(self, cell, time):
        return LineageRecord(
            id=cell.id,
            parent_id=cell.parent_id,
            root_id=cell.root_id,
            generation=cell.generation,
            birth_time=float(time),
            size_at_birth=float(cell.size_at_birth),
            shape=cell.shape,
            species=cell.species,
        )

    def register_founder(self, cell, time):
        """Add a cell with no recorded parent division (e.g. an initial cell)."""
        if cell.id in self.records:
            raise ValueError(f"Cell id {cell.id} is already in the lineage.")
        record = self._new_record(cell, time)
        self.records[cell.id] = record
        if not cell.alive:
            self.register_death(cell, time, getattr(cell, "death_cause", "killed"))
        return record

    def register_division(self, parent, daughters, time, size_at_division=None):
        """Close the parent's record as "divided" and add both daughters."""
        record = self.records[parent.id]
        record.end_time = float(time)
        record.fate = "divided"
        record.daughter_ids = tuple(d.id for d in daughters)
        if size_at_division is not None:
            record.size_at_division = float(size_at_division)
        for daughter in daughters:
            if daughter.id in self.records:
                raise ValueError(f"Cell id {daughter.id} is already in the lineage.")
            self.records[daughter.id] = self._new_record(daughter, time)

    def register_death(self, cell, time, cause):
        """Close a cell's record with fate `cause`. No-op if it already ended."""
        if cause not in FATES or cause in ("alive", "divided"):
            raise ValueError(
                f"Unknown death cause {cause!r}; must be one of "
                f"{[f for f in FATES if f not in ('alive', 'divided')]}."
            )
        record = self.records[cell.id]
        if record.end_time is not None:
            return
        record.end_time = float(time)
        record.fate = cause

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __getitem__(self, cell_id):
        return self.records[cell_id]

    def __contains__(self, cell_id):
        return cell_id in self.records

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records.values())

    def roots(self):
        """Ids of founder cells: records with no parent in this lineage."""
        return [
            cid
            for cid, rec in self.records.items()
            if rec.parent_id is None or rec.parent_id not in self.records
        ]

    def parent(self, cell_id):
        """Parent id, or None for a root."""
        parent_id = self.records[cell_id].parent_id
        return parent_id if parent_id in self.records else None

    def children(self, cell_id):
        """Daughter ids (empty unless the cell divided)."""
        return [d for d in self.records[cell_id].daughter_ids if d in self.records]

    def sisters(self, cell_id):
        """The other daughter(s) of this cell's parent."""
        parent_id = self.parent(cell_id)
        if parent_id is None:
            return []
        return [c for c in self.children(parent_id) if c != cell_id]

    def ancestors(self, cell_id):
        """Ids from this cell's parent up to its root, nearest first."""
        self.records[cell_id]  # raise KeyError for an unknown id
        chain = []
        current = self.parent(cell_id)
        while current is not None:
            chain.append(current)
            current = self.parent(current)
        return chain

    def descendants(self, cell_id):
        """Ids of every cell descended from this one (excluding itself), in DFS preorder."""
        out = []
        stack = list(reversed(self.children(cell_id)))
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(self.children(current)))
        return out

    def subtree(self, cell_id):
        """A new Lineage holding this cell and all its descendants."""
        sub = Lineage()
        sub.time = self.time
        for cid in [cell_id, *self.descendants(cell_id)]:
            rec = self.records[cid]
            sub.records[cid] = LineageRecord(**asdict(rec))
        return sub

    def leaves(self):
        """Ids of cells that did not divide (alive or dead)."""
        return [cid for cid in self.records if not self.children(cid)]

    def generation_times(self):
        """
        Series of interdivision times (end_time - birth_time), indexed by
        cell id, for every non-founder cell that divided. Founders are
        excluded: their birth_time is the simulation start, not a division.
        """
        data = {
            cid: rec.end_time - rec.birth_time
            for cid, rec in self.records.items()
            if rec.fate == "divided" and self.parent(cid) is not None
        }
        return pd.Series(data, name="generation_time", dtype=float)

    def to_dataframe(self):
        """One row per cell ever created, sorted by id."""
        columns = list(LineageRecord.__dataclass_fields__)
        rows = [asdict(self.records[cid]) for cid in sorted(self.records)]
        df = pd.DataFrame(rows, columns=columns)
        # Nullable ints, so founders' missing parent doesn't turn ids into floats.
        df["parent_id"] = df["parent_id"].astype("Int64")
        return df

    def _branch_end(self, record, t_end):
        return record.end_time if record.end_time is not None else t_end

    def to_newick(self, root_id=None, t_end=None):
        """
        Newick string of the lineage tree(s), with each branch's length
        equal to the cell's lifetime (birth to division/death, or to `t_end`
        — default `self.time` — for living cells). Nodes are labeled by
        cell id.

        With `root_id`, returns only that cell's subtree. Otherwise, a
        single founder yields one tree; several founders are joined under
        an unlabeled zero-length virtual root so the result is still one
        valid Newick tree.
        """
        t_end = self.time if t_end is None else t_end
        roots = [root_id] if root_id is not None else sorted(self.roots())
        if not roots:
            return ";"
        trees = [self._newick_subtree(r, t_end) for r in roots]
        if len(trees) == 1:
            return trees[0] + ";"
        return "(" + ",".join(trees) + ");"

    def _newick_subtree(self, root_id, t_end):
        # Iterative post-order build (no recursion limit on deep lineages).
        built = {}
        stack = [(root_id, False)]
        while stack:
            cid, expanded = stack.pop()
            kids = self.children(cid)
            if kids and not expanded:
                stack.append((cid, True))
                stack.extend((k, False) for k in reversed(kids))
                continue
            rec = self.records[cid]
            length = self._branch_end(rec, t_end) - rec.birth_time
            label = f"{cid}:{length:.6g}"
            if kids:
                label = "(" + ",".join(built.pop(k) for k in kids) + ")" + label
            built[cid] = label
        return built[root_id]

    def plot(
        self,
        ax=None,
        history=None,
        color_by=None,
        root_id=None,
        cmap="viridis",
        vmin=None,
        vmax=None,
        show_deaths=True,
    ):
        """Draw the lineage tree; see `utils.visualization.plot_lineage_tree`."""
        from ..utils.visualization import plot_lineage_tree

        return plot_lineage_tree(
            self,
            history=history,
            color_by=color_by,
            root_id=root_id,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_deaths=show_deaths,
        )
