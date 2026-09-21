"""
Benchmark the visualization pipeline against the simulation that feeds it.

Rendering a colony animation costs far more than running the simulation that
produced it, which is what limits large-population runs. This script times
each stage separately so a change can be attributed to the stage it actually
affected:

    simulate      Simulation.run (the physics, plus per-step recording)
    to_dataframe  turning the recorded history into a DataFrame
    render        _render_frames: building one RGBA image per timestep
    save_gif      encoding those images as an animated GIF
    save_mp4      encoding those images as an H.264 video
    stream_mp4    rendering and encoding in one pass, holding no frame list

Run it before and after a change, at the same --t-max, and compare:

    python examples/bench_visualization.py --t-max 9
    python examples/bench_visualization.py --t-max 9 --field

The colony is seeded from a single cell and grows exponentially, so the cell
count roughly doubles with every +1.0 of --t-max. `--t-max 9` reaches a few
hundred cells in a few seconds; `--t-max 13` reaches a few thousand and takes
considerably longer to simulate.
"""

import argparse
import os
import tempfile
import time
import tracemalloc

import matplotlib

# Force the non-interactive backend: the benchmark renders frames but never
# shows them, and an interactive backend would add windowing cost that has
# nothing to do with what's being measured.
matplotlib.use("Agg")

import numpy as np  # noqa: E402

from multicellular.core.cell import Cell  # noqa: E402
from multicellular.core.colony import Colony  # noqa: E402
from multicellular.core.environment import Environment, Field  # noqa: E402
from multicellular.core.simulation import Simulation  # noqa: E402
from multicellular.utils.visualization import _render_frames, _save_frames  # noqa: E402

GRID = 50
SIZE = 200.0


def build_simulation(t_max, dt, seed, with_field):
    """A single seed cell growing into a colony at the center of an open box."""
    rng = np.random.default_rng(seed)

    fields = None
    if with_field:
        # A static, non-diffusing gradient: enough to exercise the field
        # overlay's per-frame redraw without adding diffusion to the timings.
        ramp = np.linspace(0.0, 1.0, GRID)
        fields = [Field("nutrient", np.tile(ramp, (GRID, 1)))]

    environment = Environment(
        "bench", wall_map=np.zeros((GRID, GRID)), size=(SIZE, SIZE), fields=fields
    )
    cells = [Cell(id=0, position=(SIZE / 2, SIZE / 2), orientation=(1.0, 0.0), rng=rng)]
    return Simulation(Colony(cells, environment), dt, t_max)


def timed(label, fn, results):
    """Run `fn`, record how long it took under `label`, and return its result."""
    start = time.perf_counter()
    value = fn()
    results[label] = time.perf_counter() - start
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-max",
        type=float,
        default=9.0,
        help="Simulated duration. Cell count roughly doubles per +1.0 (default: 9).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02, help="Timestep (default: 0.02)."
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every stride-th recorded step (default: 1).",
    )
    parser.add_argument(
        "--field",
        action="store_true",
        help="Also draw a field overlay behind the cells.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Also time GIF and MP4 encoding (writes to a temporary directory).",
    )
    args = parser.parse_args()

    results = {}
    simulation = build_simulation(args.t_max, args.dt, args.seed, args.field)

    timed("simulate", lambda: simulation.run(show_progress=False), results)
    df = timed("to_dataframe", simulation.to_dataframe, results)

    env_by_time = {t: env for t, env in simulation.env_history}
    times = sorted(df["time"].unique())[:: args.stride]

    field_name = "nutrient" if args.field else None
    field_snapshots = None
    if args.field:
        field_snapshots = {t: f[field_name] for t, f in simulation.field_history}

    def render():
        # `_render_frames` yields frames; collect them here so that rendering
        # is timed on its own, separately from whatever encodes them.
        return list(
            _render_frames(
                df,
                times,
                env_by_time,
                None,
                None,
                None,
                {},
                False,
                field_name=field_name,
                field_snapshots=field_snapshots,
                field_vmax=1.0,
            )
        )

    frames = timed("render", render, results)

    sizes_mb = {}
    stream_peak_mb = None
    if args.save:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("bench.gif", "bench.mp4"):
                label = "save_" + name.split(".")[1]
                timed(
                    label,
                    lambda n=name: _save_frames(frames, tmpdir, n, fps=10.0),
                    results,
                )
                sizes_mb[name] = os.path.getsize(os.path.join(tmpdir, name)) / 1e6

            # Rendering straight into the encoder, never building a frame
            # list: this is what `visualize_colony(..., show=False)` does.
            def stream(name):
                return _save_frames(
                    _render_frames(
                        df,
                        times,
                        env_by_time,
                        None,
                        None,
                        None,
                        {},
                        False,
                        field_name=field_name,
                        field_snapshots=field_snapshots,
                        field_vmax=1.0,
                    ),
                    tmpdir,
                    name,
                    fps=10.0,
                )

            timed("stream_mp4", lambda: stream("stream.mp4"), results)

            # Measured in its own pass: tracing every allocation slows the
            # code down several-fold, which would make the timing above
            # meaningless.
            tracemalloc.start()
            stream("stream_peak.mp4")
            stream_peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()

    n_alive = int(df[df["time"] == times[-1]]["alive"].sum())
    frame_mb = frames[0].nbytes / 1e6

    print()
    print(f"t_max={args.t_max}  dt={args.dt}  stride={args.stride}  field={args.field}")
    print(
        f"{len(frames)} frames, {n_alive} cells in the final frame, "
        f"{len(df)} recorded rows"
    )
    print(
        f"frame {frames[0].shape}, {frame_mb:.2f} MB each, "
        f"{frame_mb * len(frames):.0f} MB when all held at once"
    )
    if args.save:
        print("  ".join(f"{n} {mb:.2f} MB" for n, mb in sizes_mb.items()))
        print(f"streamed render+encode peak: {stream_peak_mb:.1f} MB")
    print()
    width = max(len(label) for label in results)
    total = sum(results.values())
    for label, seconds in results.items():
        print(f"  {label:<{width}}  {seconds:8.2f}s  {100 * seconds / total:5.1f}%")
    print(f"  {'TOTAL':<{width}}  {total:8.2f}s")

    visualize = total - results["simulate"]
    print()
    print(
        f"  visualization is {visualize / results['simulate']:.1f}x the cost of "
        f"the simulation"
    )
    print(f"  {1000 * results['render'] / len(frames):.1f} ms per rendered frame")


if __name__ == "__main__":
    main()
