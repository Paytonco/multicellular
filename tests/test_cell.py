# test_cell.py

import numpy as np
import pytest

from multicellular.core.cell import Cell
from multicellular.core.reactions import Reaction, ReactionNetwork


def _make_cell(rng=None):
    """Return a cell that has reached its division target (ready to divide)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=None,
        rng=rng,
    )
    # Teleport to division target; intermediate growth doesn't affect birth-size
    # statistics and avoids thousands of tiny grow() calls in statistical tests.
    cell.length = cell._division_target
    return cell


def test_basic_cell_behavior():
    print("Initializing cell...")
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=None,
    )

    print(f"Initial volume: {cell.compute_volume():.3f}")
    print(f"Initial length: {cell.length}")
    print(f"Division target: {cell._division_target:.3f}")

    dt = 0.1
    total_time = 0.0

    print("\nGrowing...")
    while not cell.ready_to_divide():
        cell.grow(dt)
        total_time += dt
        print(
            f"t={total_time:.1f} → length={cell.length:.2f}, volume={cell.compute_volume():.2f}"
        )

    print("\nDividing...")
    daughters = cell.divide()
    if daughters:
        d1, d2 = daughters
        print("Division successful.")
        print(
            f"Daughter 1 → pos: {d1.position}, len: {d1.length:.3f}, vol: {d1.compute_volume():.2f}"
        )
        print(
            f"Daughter 2 → pos: {d2.position}, len: {d2.length:.3f}, vol: {d2.compute_volume():.2f}"
        )
    else:
        print("Division failed.")


def test_grow_dilutes_concentration_conserving_copy_number():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        growth_rate=np.log(2),
    )
    cell.set_concentration("A", 2.0)

    volume_before = cell.compute_volume()
    copy_number_before = 2.0 * volume_before

    cell.grow(dt=0.1)

    volume_after = cell.compute_volume()
    assert volume_after > volume_before  # sanity check: cell actually grew
    assert cell.concentrations["A"] * volume_after == pytest.approx(copy_number_before)
    assert cell.concentrations["A"] < 2.0  # diluted, not conserved as-is


def test_grow_with_zero_growth_rate_does_not_dilute():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        growth_rate=0.0,
    )
    cell.set_concentration("A", 2.0)

    cell.grow(dt=1.0)

    assert cell.concentrations["A"] == pytest.approx(2.0)


def test_grow_dilutes_multiple_species_independently():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        growth_rate=np.log(2),
    )
    cell.set_concentration("A", 2.0)
    cell.set_concentration("B", 5.0)

    volume_before = cell.compute_volume()
    copy_numbers_before = {"A": 2.0 * volume_before, "B": 5.0 * volume_before}

    cell.grow(dt=0.1)

    volume_after = cell.compute_volume()
    for species, copy_number in copy_numbers_before.items():
        assert cell.concentrations[species] * volume_after == pytest.approx(copy_number)


def test_step_dilutes_after_growth_with_no_reaction_network():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0,
        radius=0.5,
        network=None,
        growth_rate=np.log(2),
    )
    cell.set_concentration("A", 2.0)
    volume_before = cell.compute_volume()
    copy_number_before = 2.0 * volume_before

    cell.step(dt=0.1)

    volume_after = cell.compute_volume()
    assert cell.concentrations["A"] * volume_after == pytest.approx(copy_number_before)


def _efflux_network(k=1.0):
    rxn = Reaction(
        reactants={"X": 1},
        products={},
        exports={"X": 1},
        rate_law_type="mass_action",
        rate_params={"k": k},
    )
    return ReactionNetwork("efflux", {"R": rxn})


def test_pending_export_defaults_to_empty():
    cell = Cell(id=1, position=[0.0, 0.0], orientation=[1.0, 0.0], network=None)
    assert cell.pending_export == {}


def test_pending_export_populated_by_step_with_exporting_network():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        network=_efflux_network(),
        growth_rate=0.0,
    )
    cell.set_concentration("X", 1.0)

    cell.step(dt=0.1)

    assert cell.pending_export["X"] > 0.0


def test_pending_export_stays_empty_without_exports():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        network=ReactionNetwork(
            "noop",
            {
                "R": Reaction(
                    {"A": 1},
                    {"B": 1},
                    rate_law_type="mass_action",
                    rate_params={"k": 1.0},
                )
            },
        ),
        growth_rate=0.0,
    )
    cell.set_concentration("A", 1.0)

    cell.step(dt=0.1)

    assert cell.pending_export == {}


def test_daughters_start_with_fresh_pending_export():
    cell = _make_cell(rng=np.random.default_rng(0))
    cell.pending_export = {"X": 42.0}  # simulate a parent mid-export this step

    d1, d2 = cell.divide()

    assert d1.pending_export == {}
    assert d2.pending_export == {}


def test_lysis_targets_defaults_to_empty():
    cell = Cell(id=1, position=[0.0, 0.0], orientation=[1.0, 0.0], network=None)
    assert cell.lysis_targets == {}


def test_lysis_targets_rejects_source_equal_to_target():
    with pytest.raises(ValueError):
        Cell(
            id=1,
            position=[0.0, 0.0],
            orientation=[1.0, 0.0],
            network=None,
            lysis_targets={"X": "X"},
        )


def test_lysis_targets_propagates_to_daughters():
    cell = _make_cell(rng=np.random.default_rng(0))
    cell.lysis_targets = {"reporter_internal": "reporter_field"}

    d1, d2 = cell.divide()

    assert d1.lysis_targets == {"reporter_internal": "reporter_field"}
    assert d2.lysis_targets == {"reporter_internal": "reporter_field"}


def test_divide_conserves_concentration_by_default():
    cell = _make_cell(rng=np.random.default_rng(5))
    cell.concentrations = {"A": 2.5}

    d1, d2 = cell.divide()

    assert d1.concentrations["A"] == 2.5
    assert d2.concentrations["A"] == 2.5


def test_set_concentration_marks_low_copy_and_propagates_to_daughters():
    cell = _make_cell(rng=np.random.default_rng(0))
    cell.set_concentration("plasmid", 1.0, low_copy=True)

    assert "plasmid" in cell.low_copy_species

    d1, d2 = cell.divide()

    assert "plasmid" in d1.low_copy_species
    assert "plasmid" in d2.low_copy_species


def test_divide_low_copy_species_conserves_copy_number_binomial():
    # n = 10 <= LOW_COPY_GAUSSIAN_THRESHOLD, so the binomial branch is used.
    cell = _make_cell(rng=np.random.default_rng(0))

    parent_volume = cell.compute_volume()
    n = 10
    cell.set_concentration("plasmid", n / parent_volume, low_copy=True)

    d1, d2 = cell.divide()

    daughter_volume = d1.compute_volume()
    x1 = round(d1.concentrations["plasmid"] * daughter_volume)
    x2 = round(d2.concentrations["plasmid"] * daughter_volume)

    assert x1 + x2 == n
    assert 0 <= x1 <= n
    assert 0 <= x2 <= n


def test_divide_low_copy_species_conserves_copy_number_gaussian():
    # n > LOW_COPY_GAUSSIAN_THRESHOLD, so the Gaussian (CLT) branch is used.
    cell = _make_cell(rng=np.random.default_rng(1))

    parent_volume = cell.compute_volume()
    n = 100
    cell.set_concentration("plasmid", n / parent_volume, low_copy=True)

    d1, d2 = cell.divide()

    daughter_volume = d1.compute_volume()
    x1 = round(d1.concentrations["plasmid"] * daughter_volume)
    x2 = round(d2.concentrations["plasmid"] * daughter_volume)

    assert x1 + x2 == n
    assert 0 <= x1 <= n
    assert 0 <= x2 <= n


def test_divide_low_copy_species_mean_is_half():
    n = 100
    rng = np.random.default_rng(42)
    counts = []
    for _ in range(2000):
        cell = _make_cell(rng=rng)
        # Set concentration so copy_number = n exactly for each cell.
        cell.set_concentration("plasmid", n / cell.compute_volume(), low_copy=True)

        d1, _ = cell.divide()
        daughter_volume = d1.compute_volume()
        counts.append(round(d1.concentrations["plasmid"] * daughter_volume))

    mean = np.mean(counts)
    assert abs(mean - n / 2) < 1.0


# ---------------------------------------------------------------------------
# Adder model correctness checks (spec §CORRECTNESS CHECKS)
# ---------------------------------------------------------------------------


def _advance_to_division(cell):
    """Teleport a cell to its division target without intermediate growth steps."""
    cell.length = cell._division_target


def test_adder_mean_birth_size_converges_to_delta_bar():
    """Stationary mean birth size equals delta_bar."""
    rng = np.random.default_rng(0)
    delta_bar = 1.0
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=delta_bar,
        rng=rng,
        delta_bar=delta_bar,
        cv_delta=0.10,
        a=1.0,
        cv_f=0.0,
    )

    birth_sizes = []
    for _ in range(2000):
        _advance_to_division(cell)
        birth_sizes.append(cell.length_at_birth)
        cell = cell.divide()[0]

    mean_birth = np.mean(birth_sizes[50:])  # skip burn-in
    assert abs(mean_birth - delta_bar) < 0.05


def test_adder_birth_size_cv():
    """Stationary CV of birth size equals CV_delta / sqrt(3)."""
    rng = np.random.default_rng(1)
    cv_delta = 0.10
    delta_bar = 1.0
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=delta_bar,
        rng=rng,
        delta_bar=delta_bar,
        cv_delta=cv_delta,
        a=1.0,
        cv_f=0.0,
    )

    birth_sizes = []
    for _ in range(3000):
        _advance_to_division(cell)
        birth_sizes.append(cell.length_at_birth)
        cell = cell.divide()[0]

    birth_sizes = np.array(birth_sizes[100:])
    cv_birth = np.std(birth_sizes) / np.mean(birth_sizes)
    expected_cv = cv_delta / np.sqrt(3)
    assert abs(cv_birth - expected_cv) < 0.01


def test_adder_homeostasis():
    """
    Starting a lineage from an oversized cell, deviation from the stationary
    mean should halve exactly each generation (deterministic case, cv_delta=0).
    """
    delta_bar = 1.0
    cell = Cell(
        id=0,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=2.0 * delta_bar,  # 2× the expected stationary mean (oversized)
        rng=np.random.default_rng(99),
        delta_bar=delta_bar,
        cv_delta=0.0,  # deterministic: Delta = delta_bar every generation
        a=1.0,
        cv_f=0.0,
    )

    birth_sizes = [cell.length_at_birth]
    for _ in range(12):
        _advance_to_division(cell)
        cell = cell.divide()[0]
        birth_sizes.append(cell.length_at_birth)

    deviations = [abs(s - delta_bar) for s in birth_sizes]
    for i in range(1, len(deviations)):
        assert abs(deviations[i] - deviations[i - 1] / 2) < 1e-10


if __name__ == "__main__":
    test_basic_cell_behavior()


def _uptake_cell(growth_rate=0.0):
    uptake = Reaction(
        {"A": 1}, {"A_int": 1}, rate_law_type="mass_action", rate_params={"k": 1.0}
    )
    return Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        network=ReactionNetwork("uptake", {"R": uptake}),
        growth_rate=growth_rate,
        rng=np.random.default_rng(0),
    )


def test_step_with_external_returns_updated_external_and_conserves_mass():
    cell = _uptake_cell()
    v_ext = 20.0
    volume = cell.compute_volume()

    updated = cell.step(0.1, external={"A": 2.0}, external_volume=v_ext)

    assert set(updated) == {"A"}
    assert updated["A"] < 2.0
    assert "A" not in cell.concentrations
    taken = (2.0 - updated["A"]) * v_ext
    assert cell.copy_number("A_int") == pytest.approx(taken, rel=1e-6)
    assert cell.concentrations["A_int"] * volume == pytest.approx(taken, rel=1e-6)


def test_step_external_species_not_diluted_by_growth():
    cell = _uptake_cell(growth_rate=np.log(2))
    cell.network.reactions["R"].rate_params["k"] = 0.0

    updated = cell.step(0.5, external={"A": 2.0}, external_volume=20.0)

    assert updated["A"] == pytest.approx(2.0)


def test_step_external_requires_external_volume():
    cell = _uptake_cell()
    with pytest.raises(ValueError):
        cell.step(0.1, external={"A": 2.0})


def test_step_without_external_returns_empty_dict():
    cell = _uptake_cell()
    assert cell.step(0.1) == {}


def test_daughters_inherit_extracellular_snapshot():
    cell = _make_cell()
    cell.extracellular = {"AHL": 0.4}
    d1, d2 = cell.divide()
    assert d1.extracellular == {"AHL": 0.4} and d2.extracellular == {"AHL": 0.4}
    d1.extracellular["AHL"] = 1.0
    assert d2.extracellular["AHL"] == 0.4  # independent copies


# --- Spherical cells ---------------------------------------------------------


def _make_sphere(radius=0.5, rng=None, **kwargs):
    return Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        radius=radius,
        shape="sphere",
        rng=rng if rng is not None else np.random.default_rng(0),
        **kwargs,
    )


def test_cell_defaults_to_rod_shape():
    cell = Cell(id=1, position=[0.0, 0.0], orientation=[1.0, 0.0])
    assert cell.shape == "rod"
    assert cell.length == 2.0
    assert cell.delta_bar == 1.0


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        Cell(id=1, position=[0.0, 0.0], orientation=[1.0, 0.0], shape="cube")


def test_sphere_with_nonzero_length_raises():
    with pytest.raises(ValueError, match="length"):
        _make_sphere(length=1.0)


def test_sphere_volume_is_four_thirds_pi_r_cubed():
    cell = _make_sphere(radius=0.7)
    assert cell.length == 0.0
    assert cell.compute_volume() == pytest.approx(4.0 / 3.0 * np.pi * 0.7**3)


def test_sphere_default_delta_bar_is_initial_volume():
    cell = _make_sphere(radius=0.6)
    assert cell.delta_bar == pytest.approx(4.0 / 3.0 * np.pi * 0.6**3)
    assert cell.size_at_birth == pytest.approx(cell.compute_volume())


def test_sphere_growth_scales_radius_and_conserves_copy_number():
    cell = _make_sphere(radius=0.5, growth_rate=0.3)
    cell.set_concentration("A", 2.0)
    v0, n0 = cell.compute_volume(), cell.copy_number("A")
    cell.grow(0.5)
    assert cell.length == 0.0
    assert cell.radius == pytest.approx(0.5 * np.exp(0.3 * 0.5 / 3.0))
    assert cell.compute_volume() == pytest.approx(v0 * np.exp(0.3 * 0.5))
    assert cell.copy_number("A") == pytest.approx(n0)


def test_sphere_division_target_is_in_volume_units():
    cell = _make_sphere(radius=0.5, cv_delta=0.0, a=1.0)
    v0 = cell.compute_volume()
    assert cell._division_target == pytest.approx(2.0 * v0)
    assert not cell.ready_to_divide()
    cell.radius = 0.5 * 2.0 ** (1.0 / 3.0) * 1.0001
    assert cell.ready_to_divide()


def test_sphere_division_conserves_volume_and_daughters_touch():
    cell = _make_sphere(radius=0.5, cv_f=0.05, rng=np.random.default_rng(3))
    cell.set_concentration("A", 1.0)
    cell.set_concentration("B", 2.0, low_copy=True)
    target = cell._division_target
    cell.radius = (3.0 * target / (4.0 * np.pi)) ** (1.0 / 3.0) * (1 + 1e-9)
    n_b = int(round(cell.copy_number("B")))

    d1, d2 = cell.divide()

    for d in (d1, d2):
        assert d.shape == "sphere"
        assert d.length == 0.0
    assert d1.compute_volume() + d2.compute_volume() == pytest.approx(target)
    gap = np.linalg.norm(d2.position - d1.position)
    assert gap == pytest.approx(d1.radius + d2.radius)
    # Daughters lie along the parent's orientation.
    assert d1.position[1] == pytest.approx(0.0)
    assert d2.position[1] == pytest.approx(0.0)
    assert d1.concentrations["A"] == pytest.approx(1.0)
    assert round(d1.copy_number("B")) + round(d2.copy_number("B")) == n_b


def test_sphere_adder_mean_birth_volume_converges_to_delta_bar():
    rng = np.random.default_rng(11)
    cell = _make_sphere(radius=0.5, rng=rng, growth_rate=1.0)
    delta_bar = cell.delta_bar
    birth_sizes = []
    for _ in range(2000):
        while not cell.ready_to_divide():
            cell.grow(0.01)
        cell = cell.divide()[rng.integers(2)]
        birth_sizes.append(cell.size_at_birth)
    assert np.mean(birth_sizes[200:]) == pytest.approx(delta_bar, rel=0.05)


def test_to_dict_includes_shape():
    assert _make_sphere().to_dict()["shape"] == "sphere"
