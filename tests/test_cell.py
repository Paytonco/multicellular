# test_cell.py

import numpy as np

from multicellular.core.cell import Cell


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

    dt = 0.1
    total_time = 0.0
    growth_rate = 1.0  # Units: length per unit time

    print("\nGrowing...")
    while not cell.ready_to_divide(threshold_length=4.0):
        cell.grow(dt, growth_rate=growth_rate)
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
            f"Daughter 1 → pos: {d1.position}, len: {d1.length}, vol: {d1.compute_volume():.2f}"
        )
        print(
            f"Daughter 2 → pos: {d2.position}, len: {d2.length}, vol: {d2.compute_volume():.2f}"
        )
    else:
        print("Division failed.")


def test_divide_conserves_concentration_by_default():
    cell = Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=4.0,
        radius=0.5,
        network=None,
    )
    cell.concentrations = {"A": 2.5}

    d1, d2 = cell.divide()

    assert d1.concentrations["A"] == 2.5
    assert d2.concentrations["A"] == 2.5


def _make_cell(rng=None):
    return Cell(
        id=1,
        position=[0.0, 0.0],
        orientation=[1.0, 0.0],
        length=4.0,
        radius=0.5,
        network=None,
        rng=rng,
    )


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
    parent_volume = _make_cell().compute_volume()

    n = 100
    rng = np.random.default_rng(42)
    counts = []
    for _ in range(2000):
        cell = _make_cell(rng=rng)
        cell.set_concentration("plasmid", n / parent_volume, low_copy=True)

        d1, _ = cell.divide()
        daughter_volume = d1.compute_volume()
        counts.append(round(d1.concentrations["plasmid"] * daughter_volume))

    mean = np.mean(counts)
    assert abs(mean - n / 2) < 1.0


if __name__ == "__main__":
    test_basic_cell_behavior()
