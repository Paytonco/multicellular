# test_cell.py

import numpy as np

from multicellular.core.cell import Cell


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
