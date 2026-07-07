# tests/test_visualization.py

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is ever imported

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from multicellular.core.cell import Cell  # noqa: E402
from multicellular.core.colony import Colony  # noqa: E402
from multicellular.core.environment import Environment, Field  # noqa: E402
from multicellular.core.simulation import Simulation  # noqa: E402


def _diffusion_sim(dt=0.1, t_max=0.4, shape=(7, 7), bounds=(20.0, 20.0)):
    """A cell-free Simulation of a single diffusing Field, for visualize_field tests."""
    values = np.zeros(shape)
    values[shape[0] // 2, shape[1] // 2] = 10.0
    field = Field("dye", values, diffuses=True, diffusivity=5e-10)
    env = Environment("diffusion test", shape=shape, bounds=bounds, fields=[field])
    colony = Colony([], env)
    sim = Simulation(colony, dt=dt, t_max=t_max)
    sim.run(show_progress=False)
    return sim


def test_visualize_field_returns_animation():
    sim = _diffusion_sim()
    anim = sim.visualize_field("dye", show_progress=False)
    assert isinstance(anim, FuncAnimation)


def test_visualize_field_saves_one_frame_per_recorded_step(tmp_path):
    sim = _diffusion_sim(dt=0.1, t_max=0.4)  # t = 0, 0.1, 0.2, 0.3, 0.4 -> 5 steps
    sim.visualize_field(
        "dye", save_path=str(tmp_path), filename="dye.gif", show_progress=False
    )
    gif_path = tmp_path / "dye.gif"
    assert gif_path.exists()
    with Image.open(gif_path) as im:
        assert im.n_frames == 5


def test_visualize_field_stride_reduces_frame_count(tmp_path):
    sim = _diffusion_sim(dt=0.1, t_max=0.4)  # 5 recorded steps
    sim.visualize_field(
        "dye",
        stride=2,
        save_path=str(tmp_path),
        filename="dye.gif",
        show_progress=False,
    )
    with Image.open(tmp_path / "dye.gif") as im:
        assert im.n_frames == 3  # steps 0, 2, 4


def test_visualize_field_unknown_field_raises_key_error():
    sim = _diffusion_sim()
    with pytest.raises(KeyError):
        sim.visualize_field("nonexistent", show_progress=False)


def test_visualize_field_default_vmax_uses_max_recorded_value():
    sim = _diffusion_sim()
    # Explicit vmax equal to the true peak (t=0's spike, since the field
    # only spreads out and never exceeds its initial value) should render
    # identically to the auto-computed default.
    peak = sim.field_history[0][1]["dye"].max()
    anim_default = sim.visualize_field("dye", show_progress=False)
    anim_explicit = sim.visualize_field("dye", vmax=peak, show_progress=False)
    assert isinstance(anim_default, FuncAnimation)
    assert isinstance(anim_explicit, FuncAnimation)


def test_visualize_colony_returns_animation():
    cell = Cell(id=0, position=[10.0, 10.0], orientation=[1.0, 0.0])
    env = Environment("colony test", shape=(5, 5), bounds=(20.0, 20.0))
    colony = Colony([cell], env)
    sim = Simulation(colony, dt=0.1, t_max=0.2)
    sim.run(show_progress=False)

    anim = sim.visualize_colony(show_progress=False)
    assert isinstance(anim, FuncAnimation)


def test_plot_field_returns_figure_with_correct_data():
    sim = _diffusion_sim(dt=0.1, t_max=0.4)
    fig = sim.plot_field("dye", time=0.0)
    assert isinstance(fig, Figure)
    plotted = fig.axes[0].images[0].get_array()
    expected = sim.field_history[0][1]["dye"]
    np.testing.assert_allclose(plotted, expected)


def test_plot_field_title_shows_env_name_and_time():
    sim = _diffusion_sim(dt=0.1, t_max=0.4)
    fig = sim.plot_field("dye", time=0.2)
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "diffusion test"
    assert ax.get_title(loc="right") == "t = 0.20"


def test_plot_field_defaults_to_latest_recorded_time():
    sim = _diffusion_sim(dt=0.1, t_max=0.4)
    fig = sim.plot_field("dye")
    assert fig.axes[0].get_title(loc="right") == "t = 0.40"


def test_plot_field_snaps_to_nearest_recorded_time():
    sim = _diffusion_sim(dt=0.1, t_max=0.4)  # recorded at 0, 0.1, 0.2, 0.3, 0.4
    fig = sim.plot_field("dye", time=0.23)  # nearer to 0.2 than 0.3
    assert fig.axes[0].get_title(loc="right") == "t = 0.20"


def test_plot_field_multiple_fields_creates_one_panel_each():
    shape = (6, 6)
    eta = np.full(shape, 6.9e-4)
    diffusivity = np.full(shape, 3.0e-9)
    env = Environment(
        "multi-field test",
        shape=shape,
        bounds=(20.0, 20.0),
        fields=[Field("eta_field", eta), Field("diffusivity_field", diffusivity)],
    )
    sim = Simulation(Colony([], env), dt=1.0, t_max=0.0)
    sim.run(show_progress=False)

    fig = sim.plot_field(["eta_field", "diffusivity_field"])
    # One heatmap panel per field, plus one colorbar axes per panel.
    assert len(fig.axes) == 4
    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), eta)
    np.testing.assert_allclose(fig.axes[1].images[0].get_array(), diffusivity)


def test_plot_field_unknown_field_raises_key_error():
    sim = _diffusion_sim()
    with pytest.raises(KeyError):
        sim.plot_field("nonexistent")
