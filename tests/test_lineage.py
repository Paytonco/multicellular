# tests/test_lineage.py

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is ever imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from multicellular.core.cell import Cell  # noqa: E402
from multicellular.core.colony import Colony  # noqa: E402
from multicellular.core.environment import Environment  # noqa: E402
from multicellular.core.lineage import Lineage, LineageRecord  # noqa: E402
from multicellular.core.reactions import Reaction, ReactionNetwork  # noqa: E402
from multicellular.core.simulation import Simulation  # noqa: E402


def _env(n=20, size=(20.0, 20.0)):
    return Environment("env", wall_map=np.zeros((n, n)), size=size)


def _cell(id=None, position=(10.0, 10.0), seed=0, **kw):
    return Cell(
        id=id,
        position=list(position),
        orientation=[1.0, 0.0],
        length=2.0,
        rng=np.random.default_rng(seed),
        **kw,
    )


def _ready_to_divide(cell):
    cell.length = cell._division_target
    return cell


def _hand_built_lineage():
    """
    0 ─┬─ 1 ─┬─ 3
       │     └─ 4
       └─ 2
    5            (second founder)
    """
    lin = Lineage()
    lin.time = 10.0

    def rec(i, parent, root, gen, birth, end=None, fate="alive", kids=()):
        lin.records[i] = LineageRecord(
            id=i,
            parent_id=parent,
            root_id=root,
            generation=gen,
            birth_time=birth,
            end_time=end,
            fate=fate,
            daughter_ids=kids,
        )

    rec(0, None, 0, 0, 0.0, 2.0, "divided", (1, 2))
    rec(1, 0, 0, 1, 2.0, 5.0, "divided", (3, 4))
    rec(2, 0, 0, 1, 2.0, 7.0, "survival_condition")
    rec(3, 1, 0, 2, 5.0)
    rec(4, 1, 0, 2, 5.0, 6.0, "out_of_bounds")
    rec(5, None, 5, 0, 0.0)
    return lin


# ---------------------------------------------------------------------------
# Ids and founders
# ---------------------------------------------------------------------------


def test_colony_assigns_unique_ids_to_id_less_cells():
    cells = [
        _cell(id=None),
        _cell(id=7, position=(5, 5)),
        _cell(id=None, position=(15, 15)),
    ]
    colony = Colony(cells, _env())
    ids = [c.id for c in colony.cells]
    assert len(set(ids)) == 3
    assert 7 in ids
    assert all(i is not None for i in ids)


def test_colony_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        Colony([_cell(id=1), _cell(id=1, position=(5, 5))], _env())


def test_founders_are_their_own_roots():
    colony = Colony([_cell(id=3), _cell(id=None, position=(5, 5))], _env())
    for cell in colony.cells:
        assert cell.root_id == cell.id
        assert cell.parent_id is None
        assert cell.generation == 0
        rec = colony.lineage[cell.id]
        assert rec.birth_time == 0.0
        assert rec.fate == "alive"
        assert rec.end_time is None
        assert rec.size_at_birth == pytest.approx(cell.size_at_birth)
    assert sorted(colony.lineage.roots()) == sorted(c.id for c in colony.cells)


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------


def test_division_links_daughters_to_parent():
    parent = _ready_to_divide(_cell(id=0))
    target = parent._division_target
    colony = Colony([parent], _env())
    colony.step(dt=0.1)

    d1, d2 = colony.cells
    for d in (d1, d2):
        assert d.parent_id == 0
        assert d.generation == 1
        assert d.root_id == 0
        rec = colony.lineage[d.id]
        assert rec.birth_time == pytest.approx(0.1)
        assert rec.parent_id == 0

    prec = colony.lineage[0]
    assert prec.fate == "divided"
    assert prec.end_time == pytest.approx(0.1)
    assert prec.daughter_ids == (d1.id, d2.id)
    assert prec.size_at_division == pytest.approx(target)
    assert colony.lineage.children(0) == [d1.id, d2.id]
    assert colony.lineage.sisters(d1.id) == [d2.id]
    assert colony.time == pytest.approx(0.1)


def test_generation_and_root_propagate_over_several_divisions():
    colony = Colony([_cell(id=0, seed=1)], _env(n=40, size=(80.0, 80.0)))
    for _ in range(400):
        colony.step(dt=0.01)
    lin = colony.lineage
    assert len(lin) > 3  # at least two rounds of division
    for rec in lin:
        assert rec.root_id == 0
        if rec.parent_id is not None:
            parent = lin[rec.parent_id]
            assert rec.generation == parent.generation + 1
            assert rec.birth_time == parent.end_time
            assert rec.id in parent.daughter_ids
    n_divided = sum(rec.fate == "divided" for rec in lin)
    assert len(lin) - len(lin.roots()) == 2 * n_divided


# ---------------------------------------------------------------------------
# Death causes
# ---------------------------------------------------------------------------


def test_out_of_bounds_death_is_recorded():
    inside = _cell(id=0, position=(10, 10))
    outside = _cell(id=1, position=(30, 10))
    colony = Colony([inside, outside], _env())
    colony.step(dt=0.1)
    rec = colony.lineage[1]
    assert rec.fate == "out_of_bounds"
    assert rec.end_time == pytest.approx(0.1)
    assert colony.lineage[0].fate == "alive"


def test_survival_condition_death_is_recorded():
    colony = Colony([_cell(id=0)], _env(), survival_conditions=[("A", ">", 0)])
    colony.step(dt=0.1)
    rec = colony.lineage[0]
    assert rec.fate == "survival_condition"
    assert rec.end_time == pytest.approx(0.1)


def test_manual_kill_is_recorded_at_the_current_time_and_only_once():
    colony = Colony([_cell(id=0)], _env())
    colony.step(dt=0.1)
    colony.cells[0].kill()
    colony.step(dt=0.1)
    colony.step(dt=0.1)
    rec = colony.lineage[0]
    assert rec.fate == "killed"
    assert rec.end_time == pytest.approx(0.1)


def test_cells_appended_after_construction_are_registered_as_founders():
    colony = Colony([_cell(id=0)], _env())
    colony.step(dt=0.1)
    newcomer = _cell(id=None, position=(5, 5))
    colony.cells.append(newcomer)
    colony.step(dt=0.1)
    assert newcomer.id is not None and newcomer.id != 0
    rec = colony.lineage[newcomer.id]
    assert rec.parent_id is None
    assert rec.root_id == newcomer.id
    assert rec.birth_time == pytest.approx(0.1)


def test_register_death_rejects_unknown_cause():
    colony = Colony([_cell(id=0)], _env())
    with pytest.raises(ValueError, match="death cause"):
        colony.lineage.register_death(colony.cells[0], 0.0, "divided")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def test_tree_queries_on_hand_built_lineage():
    lin = _hand_built_lineage()
    assert sorted(lin.roots()) == [0, 5]
    assert lin.parent(3) == 1
    assert lin.parent(0) is None
    assert lin.children(0) == [1, 2]
    assert lin.children(2) == []
    assert lin.sisters(1) == [2]
    assert lin.sisters(0) == []
    assert lin.ancestors(3) == [1, 0]
    assert lin.ancestors(0) == []
    assert lin.descendants(0) == [1, 3, 4, 2]
    assert sorted(lin.leaves()) == [2, 3, 4, 5]

    sub = lin.subtree(1)
    assert sorted(sub.records) == [1, 3, 4]
    assert sub.roots() == [1]
    assert sub.time == lin.time

    # Only non-founder cells that divided: cell 1 (born 2, divided 5).
    gt = lin.generation_times()
    assert gt.to_dict() == {1: 3.0}


def test_ancestors_raises_for_unknown_id():
    with pytest.raises(KeyError):
        _hand_built_lineage().ancestors(99)


def test_to_dataframe_has_one_row_per_cell():
    df = _hand_built_lineage().to_dataframe()
    assert list(df["id"]) == [0, 1, 2, 3, 4, 5]
    for col in ("parent_id", "root_id", "generation", "birth_time", "end_time", "fate"):
        assert col in df.columns
    assert df.set_index("id").loc[4, "fate"] == "out_of_bounds"


def test_to_newick_single_tree_and_branch_lengths():
    lin = _hand_built_lineage()
    # Living leaf 3 runs to lineage.time (10): root-to-tip 2 + 3 + 5 = 10.
    assert lin.to_newick(root_id=0) == "((3:5,4:1)1:3,2:5)0:2;"


def test_to_newick_forest_uses_virtual_root_and_t_end():
    lin = _hand_built_lineage()
    assert lin.to_newick() == "(((3:5,4:1)1:3,2:5)0:2,5:10);"
    assert lin.to_newick(t_end=12.0) == "(((3:7,4:1)1:3,2:5)0:2,5:12);"


def test_to_newick_handles_deep_lineages_without_recursion():
    lin = Lineage()
    depth = 5000
    for i in range(depth):
        lin.records[i] = LineageRecord(
            id=i,
            parent_id=i - 1 if i else None,
            root_id=0,
            generation=i,
            birth_time=float(i),
            end_time=float(i + 1),
            fate="divided" if i < depth - 1 else "killed",
            daughter_ids=(i + 1,) if i < depth - 1 else (),
        )
    assert lin.to_newick().endswith("0:1;")


# ---------------------------------------------------------------------------
# Simulation integration
# ---------------------------------------------------------------------------


def _growing_sim(t_max=3.0, dt=0.05):
    production = Reaction(
        {}, {"X": 1}, rate_law_type="mass_action", rate_params={"k": 1.0}
    )
    network = ReactionNetwork("production", {"R": production})
    cell = _cell(id=None, network=network, seed=2)
    colony = Colony([cell], _env(n=40, size=(60.0, 60.0)), brownian_motion=False)
    sim = Simulation(colony, dt=dt, t_max=t_max)
    sim.run(show_progress=False)
    return sim


@pytest.fixture(scope="module")
def grown_sim():
    return _growing_sim()


def test_history_has_lineage_columns(grown_sim):
    df = grown_sim.to_dataframe()
    for col in ("parent_id", "generation", "root_id"):
        assert col in df.columns
    assert df["generation"].max() >= 1


def test_lineage_birth_times_match_first_appearance_in_history(grown_sim):
    df = grown_sim.to_dataframe()
    first_seen = df.groupby("cell_id")["time"].min()
    lin_df = grown_sim.lineage_dataframe().set_index("id")
    assert set(first_seen.index) == set(lin_df.index)
    for cid, t in first_seen.items():
        assert lin_df.loc[cid, "birth_time"] == pytest.approx(t, abs=1e-12)
    divided = lin_df[lin_df["fate"] == "divided"]
    assert len(divided) >= 1
    # A divided parent's last recorded time is one step before it ended.
    for cid, rec in divided.iterrows():
        last = df.loc[df["cell_id"] == cid, "time"].max()
        assert rec["end_time"] == pytest.approx(last + grown_sim.dt)
    assert grown_sim.lineage.time == pytest.approx(grown_sim.time)


def _line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def test_plot_lineage_without_color(grown_sim):
    fig, ax = grown_sim.plot_lineage()
    lin = grown_sim.lineage
    branches, connectors = _line_collections(ax)
    assert len(branches.get_segments()) == len(lin)
    n_divided = sum(rec.fate == "divided" for rec in lin)
    assert len(connectors.get_segments()) == n_divided
    plt.close(fig)


def test_plot_lineage_colored_by_species(grown_sim):
    fig, ax = grown_sim.plot_lineage(color_by="X")
    branches = _line_collections(ax)[0]
    assert branches.get_array() is not None
    assert len(branches.get_segments()) >= len(grown_sim.lineage)
    assert len(fig.axes) == 2  # plot + colorbar
    plt.close(fig)


def test_plot_lineage_marks_deaths_and_accepts_root_id():
    lin = _hand_built_lineage()
    fig, ax = lin.plot(root_id=0)
    branches, connectors = _line_collections(ax)
    assert len(branches.get_segments()) == 5  # cells 0-4, not founder 5
    assert len(connectors.get_segments()) == 2
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {"survival condition", "out of bounds"}
    plt.close(fig)


def test_plot_lineage_color_by_requires_history():
    with pytest.raises(ValueError, match="history"):
        _hand_built_lineage().plot(color_by="X")
