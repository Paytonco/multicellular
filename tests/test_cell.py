# test_cell.py

import numpy as np

from core.cell import Cell


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


if __name__ == "__main__":
    test_basic_cell_behavior()
