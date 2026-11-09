# utils/visualization.py

import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

# Save filenames ending in one of these are encoded as video (via imageio,
# which bundles its own ffmpeg through imageio-ffmpeg, so no system ffmpeg is
# needed); anything else is written as an animated GIF by Pillow.
_VIDEO_EXTENSIONS = {".mp4"}

# Gray level (in every RGB channel) of cells when no channel is tied to a
# chemical species. Once any channel is in use, unused channels are 0 instead,
# so a cell colored only by `red` is pure red.
_DEFAULT_CHANNEL_VALUE = 0.3

# Transparency and stacking order for the optional field heatmap drawn
# behind cells in `_render_frames`: below the cell patches (zorder=2) but
# above the wall_map background (zorder=1), so it reads as a subtle
# backdrop rather than competing with the cells for attention.
_FIELD_OVERLAY_ALPHA = 0.45
_FIELD_OVERLAY_ZORDER = 1.5

# Cell outlines in `_render_frames`: mobile cells get a thin black edge,
# anchored (fixed) cells a heavier slate-blue one, which stays distinct from
# any fill color (gray default or species-driven RGB).
_MOBILE_EDGE_COLOR = (0.0, 0.0, 0.0, 1.0)
_ANCHORED_EDGE_COLOR = (0.2, 0.35, 0.75, 1.0)
_ANCHORED_LINEWIDTH = 2.5

# wall_map entry -> RGB, used to paint the environment background: media (0)
# is left white/uncolored so it doesn't compete with the field overlay, wall
# (1) is a solid neutral gray, out-of-bounds (-1) is the same red tint as the
# margin beyond the environment's physical extent.
_WALL_MAP_COLORS = {
    -1: (0.86, 0.24, 0.24),
    0: (1.0, 1.0, 1.0),
    1: (0.35, 0.35, 0.35),
}


def _wall_map_rgba(wall_map):
    """Render a wall_map matrix as an RGBA image per `_WALL_MAP_COLORS`."""
    rgba = np.ones(wall_map.shape + (4,))
    for value, color in _WALL_MAP_COLORS.items():
        rgba[wall_map == value, 0:3] = color
    return rgba


def _cap_template(n_points):
    """
    The unit capsule outline, split into the parts that scale with a cell's
    length and with its radius.

    Returns (axial_sign, cos_angles, sin_angles), each of length
    `2 * n_points`, such that a cell's outline in its own frame is
        x = axial_sign * (length / 2) + radius * cos_angles
        y =                             radius * sin_angles
    The template depends only on `n_points`, so it is computed once per
    render rather than once per cell.
    """
    right_angles = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, n_points)
    angles = np.concatenate([right_angles, left_angles])
    axial_sign = np.concatenate([np.ones(n_points), -np.ones(n_points)])
    return axial_sign, np.cos(angles), np.sin(angles)


def _capsule_outlines(positions, orientations, lengths, radii, n_points=12):
    """
    Return the (x, y) outlines of many rod-shaped (capsule) cells at once, as
    an array of shape (n_cells, 2 * n_points, 2).

    Each cell is a cylinder of the given length and radius, centered at its
    position and aligned with its orientation, with hemispherical caps at
    each end — the same shape `_capsule_outline` produces for a single cell,
    but built for every cell in one set of array operations. Rendering a
    frame touches every cell, so doing this per cell in Python is a large
    part of what made animating a big colony slow.

    The rotation is applied directly from the orientation vector's components
    rather than via `arctan2` followed by `cos`/`sin`: for a unit orientation
    (ox, oy), cos(theta) = ox and sin(theta) = oy by definition.
    """
    axial_sign, cos_angles, sin_angles = _cap_template(n_points)

    half_lengths = np.asarray(lengths, dtype=float)[:, None] / 2.0
    radii = np.asarray(radii, dtype=float)[:, None]
    local_x = axial_sign * half_lengths + radii * cos_angles
    local_y = radii * sin_angles

    orientations = np.asarray(orientations, dtype=float)
    norms = np.hypot(orientations[:, 0], orientations[:, 1])
    # A zero-length orientation has no defined angle; leave such a cell
    # unrotated instead of dividing by zero.
    norms[norms == 0.0] = 1.0
    cos_theta = (orientations[:, 0] / norms)[:, None]
    sin_theta = (orientations[:, 1] / norms)[:, None]

    positions = np.asarray(positions, dtype=float)
    x = local_x * cos_theta - local_y * sin_theta + positions[:, 0:1]
    y = local_x * sin_theta + local_y * cos_theta + positions[:, 1:2]
    return np.stack([x, y], axis=-1)


def _capsule_outline(position, orientation, length, radius, n_points=12):
    """
    Return the (x, y) outline of a single rod-shaped (capsule) cell.

    A convenience wrapper around `_capsule_outlines`, which does the same
    work for a whole population at once.
    """
    return _capsule_outlines(
        [position], [orientation], [length], [radius], n_points=n_points
    )[0]


def _channel_column(df, species, scale):
    """
    The per-cell value of one RGB channel for every row of `df`, normalized
    to [0, 1] by `scale`.

    The array counterpart of reading one species' concentration out of a
    single row: a channel not tied to a species, or a species with no column
    (or a non-positive scale), contributes nothing. Rows where the species
    is absent — a colony whose cells carry different networks gives those
    rows NaN — are treated as zero concentration.
    """
    if species is None or species not in df.columns or scale <= 0:
        return np.zeros(len(df))
    values = np.nan_to_num(df[species].to_numpy(dtype=float))
    return np.clip(values / scale, 0.0, 1.0)


def _add_field_overlay(ax, field_name, values, width, height, cmap, vmin, vmax):
    """
    Draw a Field's values as a light, semi-transparent heatmap layer behind
    the cells, and add a colorbar labeled with the field's name.

    The overlay's extent is exactly the environment rectangle (0 to width, 0
    to height) that `_render_frames` paints its wall_map background onto —
    the same region the red out-of-bounds tint is painted *outside* of — so
    the overlay sits on top of that background within bounds and never
    bleeds into the tinted margin. `zorder` places it above the wall_map
    background but below the cell patches (see `_FIELD_OVERLAY_ZORDER`).
    """
    image = ax.imshow(
        values,
        origin="lower",
        extent=[0, width, 0, height],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=_FIELD_OVERLAY_ALPHA,
        zorder=_FIELD_OVERLAY_ZORDER,
    )
    ax.figure.colorbar(image, ax=ax, label=field_name)
    return image


def _prepare_blitting(fig, ax, dynamic_artists):
    """
    Draw `fig` once and cache everything in it that never changes, so that
    later frames only have to redraw the artists that do.

    Most of what an animation frame contains is identical in every frame: the
    axes, ticks, labels, colorbar, and (for a colony) the red out-of-bounds
    margin and the wall_map image. Repainting all of it per frame dominates
    render time once the per-cell cost has been dealt with, so it is painted
    once here and restored from the cache per frame instead.

    Returns `(background, draw_order)`: the cached background to restore, and
    the artists to redraw over it, in the order they must be drawn. Drawing by
    hand bypasses the zorder sort a full redraw does, so `draw_order` is
    `dynamic_artists` (which the caller passes in zorder order) followed by
    the axes spines. The spines are static, but a full redraw paints them
    *after* the axes contents — so a changing image, whose edge covers the
    inner half of the spine line, has to be drawn under them here too, or the
    frame ends up with visibly thinner axes edges.

    Everything in `draw_order` is kept out of the one draw that produces the
    background, so the cache holds only what stays put; otherwise each frame
    would be composited on top of the first frame's contents. Which mechanism
    does that has to differ by artist type, and both halves are load-bearing:

    - Images are hidden, because `animated` does not exclude them. `Axes.draw`
      deliberately keeps animated `AxesImage`s in its draw list (see
      `axes/_base.py`), so a semi-transparent field overlay marked `animated`
      would be blended into the background *and* again into every frame,
      coming out twice as strong as it should be.
    - Everything else is marked `animated` rather than hidden, because a
      hidden title is left misplaced: `Axes.draw` recomputes title positions
      from what is visible, and an invisible title ends up pushed above the
      top of the figure, where it is then never seen again.

    `background` is None if the backend cannot blit (the pure-vector
    backends, e.g. pdf/svg, have no pixel buffer to copy), in which case
    `_blit_frame` falls back to redrawing the whole figure.
    """
    canvas = fig.canvas
    can_blit = hasattr(canvas, "copy_from_bbox") and hasattr(canvas, "restore_region")

    draw_order = list(dynamic_artists) + list(ax.spines.values())
    if not can_blit:
        canvas.draw()
        return None, draw_order

    for artist in draw_order:
        if isinstance(artist, AxesImage):
            artist.set_visible(False)
        else:
            artist.set_animated(True)

    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)

    for artist in draw_order:
        artist.set_visible(True)

    return background, draw_order


def _blit_frame(fig, ax, draw_order, background):
    """
    Composite one frame and return it as an RGBA array.

    Restores the cached static background, then redraws just the artists in
    `draw_order` (see `_prepare_blitting`). With no cached background,
    redraws the whole figure instead.
    """
    canvas = fig.canvas
    if background is None:
        canvas.draw()
    else:
        canvas.restore_region(background)
        for artist in draw_order:
            ax.draw_artist(artist)
    return np.asarray(canvas.buffer_rgba()).copy()


def _cell_colors(df, red, green, blue, scales):
    """
    The (n_rows, 3) RGB color of every row of `df`.

    Each channel named by a species is that species' normalized
    concentration (see `_channel_column`), and unused channels are 0, so a
    cell colored only by `red` is pure red. With no channel in use, every
    cell is a uniform gray (`_DEFAULT_CHANNEL_VALUE` in all three channels).
    """
    if red is None and green is None and blue is None:
        return np.full((len(df), 3), _DEFAULT_CHANNEL_VALUE)
    return np.column_stack(
        [
            _channel_column(df, species, scales.get(species, 1.0))
            for species in (red, green, blue)
        ]
    )


def _render_frames(
    df,
    times,
    env_by_time,
    red,
    green,
    blue,
    scales,
    show_progress,
    field_name=None,
    field_snapshots=None,
    field_cmap="YlOrRd",
    field_vmin=0.0,
    field_vmax=None,
    n_points=12,
):
    """
    Yield every animation frame as an RGBA image, one at a time.

    Rendering is done here, rather than live during playback, so displaying
    (or saving) the animation afterward is just fast image blitting,
    regardless of how large the colony grows. Frames are yielded rather than
    returned as a list so that a caller writing them to a file can encode
    each one and let it go: a long animation is hundreds of megabytes of RGBA
    if it is all held at once.

    All of a frame's cells are drawn as a single `PolyCollection` whose
    vertices and colors are recomputed per frame, rather than as one
    `Polygon` patch per cell. Matplotlib's cost is dominated by the number of
    artists it has to manage and draw, so one collection of N polygons is far
    cheaper than N separate patches — which is what made animating a large
    colony slow. For the same reason every cell's outline and color is
    computed for the whole run up front, in array operations, instead of row
    by row inside the loop.

    `n_points` is the number of vertices per hemispherical cap; lowering it
    cuts per-cell geometry cost for colonies whose cells are only a few pixels
    across.
    """
    first_env = next(iter(env_by_time.values()))
    width, height = first_env.size
    pad = max(width, height) * 0.1
    x_min = min(0.0, df["position_x"].min()) - pad
    x_max = max(width, df["position_x"].max()) + pad
    y_min = min(0.0, df["position_y"].min()) - pad
    y_max = max(height, df["position_y"].max()) + pad

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Tint the margin beyond the environment's physical extent red (out of
    # bounds), then paint the environment itself from its wall_map: media
    # white, walls gray, and any interior out-of-bounds (-1) cells the same
    # red as the margin.
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="red",
            alpha=0.2,
            zorder=0,
        )
    )
    ax.imshow(
        _wall_map_rgba(first_env.wall_map),
        origin="lower",
        extent=[0, width, 0, height],
        zorder=1,
        interpolation="nearest",
    )

    field_image = None
    if field_name is not None:
        field_image = _add_field_overlay(
            ax,
            field_name,
            field_snapshots[times[0]],
            width,
            height,
            field_cmap,
            field_vmin,
            field_vmax,
        )

    # Every living cell's outline and color, for every recorded time, computed
    # in one pass. Sorted by time so each frame's cells are one contiguous
    # slice, located with `searchsorted` below: `times` values come straight
    # out of this same column, so they compare exactly.
    alive = df[df["alive"]].sort_values("time", kind="stable")
    outlines = _capsule_outlines(
        alive[["position_x", "position_y"]].to_numpy(dtype=float),
        alive[["orientation_x", "orientation_y"]].to_numpy(dtype=float),
        alive["length"].to_numpy(dtype=float),
        alive["radius"].to_numpy(dtype=float),
        n_points=n_points,
    )
    colors = _cell_colors(alive, red, green, blue, scales)
    # Anchored cells get a heavier, slate-blue outline so fixed host cells
    # stand out from the mobile population. Histories recorded before the
    # `anchored` column existed have none: every cell is mobile.
    if "anchored" in alive.columns:
        anchored = alive["anchored"].fillna(False).to_numpy(dtype=bool)
    else:
        anchored = np.zeros(len(alive), dtype=bool)
    any_anchored = bool(anchored.any())
    if any_anchored:
        edgecolors = np.where(
            anchored[:, None],
            np.array(_ANCHORED_EDGE_COLOR),
            np.array(_MOBILE_EDGE_COLOR),
        )
        linewidths = np.where(
            anchored, _ANCHORED_LINEWIDTH, plt.rcParams["patch.linewidth"]
        )
    alive_times = alive["time"].to_numpy(dtype=float)
    frame_starts = np.searchsorted(alive_times, times, side="left")
    frame_stops = np.searchsorted(alive_times, times, side="right")

    # One artist for the whole population, reused across frames: only its
    # vertices and face colors change from frame to frame.
    #
    # Two details keep the output pixel-identical to the per-patch drawing
    # this replaced. `joinstyle`: a Collection defaults to round joins, a
    # Polygon patch to mitered ones. `antialiaseds`: given a single path and
    # one of everything else, `Collection.draw` switches to a marker-stamping
    # fast path (`draw_markers`) that positions the stamp up to a pixel off
    # from where `draw_path` would put it — so a one-cell frame would render
    # differently from every other frame. Passing two (identical) flags fails
    # that all-length-one test and keeps every frame on the same code path.
    cells = PolyCollection(
        [],
        edgecolors="black",
        zorder=2,
        joinstyle="miter",
        antialiaseds=[plt.rcParams["patch.antialiased"]] * 2,
    )
    ax.add_collection(cells, autolim=False)

    # Created empty, and filled in per frame below: the titles change with the
    # time (and the environment), so they are redrawn per frame like the cells
    # rather than baked into the cached background. `set_title` hands back the
    # Text artist that holds them.
    left_title = ax.set_title("", loc="left")
    right_title = ax.set_title("", loc="right")

    # In zorder order, since `_blit_frame` draws them in the order given.
    dynamic = [
        a for a in (field_image, cells, left_title, right_title) if a is not None
    ]
    background, draw_order = _prepare_blitting(fig, ax, dynamic)

    try:
        for index, t in enumerate(
            tqdm(times, disable=not show_progress, desc="Rendering frames")
        ):
            if field_image is not None:
                field_image.set_data(field_snapshots[t])

            start, stop = frame_starts[index], frame_stops[index]
            cells.set_verts(outlines[start:stop])
            cells.set_facecolor(colors[start:stop])
            if any_anchored:
                cells.set_edgecolor(edgecolors[start:stop])
                cells.set_linewidth(linewidths[start:stop])

            env = env_by_time.get(t, first_env)
            left_title.set_text(env.name)
            right_title.set_text(f"t = {t:.2f}")

            yield _blit_frame(fig, ax, draw_order, background)
    finally:
        plt.close(fig)


def _render_field_frames(
    times, env_by_time, snapshots, field_name, cmap, vmin, vmax, show_progress
):
    """
    Yield every field-animation frame as an RGBA image, one at a time — the
    same strategy `_render_frames` uses for cell colonies.

    Only the heatmap and the titles differ between frames, so — as in
    `_render_frames` — the rest of the figure is drawn once and restored from
    a cached background per frame (see `_prepare_blitting`).
    """
    first_env = next(iter(env_by_time.values()))
    width, height = first_env.size

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    image = ax.imshow(
        snapshots[times[0]],
        origin="lower",
        extent=[0, width, 0, height],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, label=field_name)

    left_title = ax.set_title("", loc="left")
    right_title = ax.set_title("", loc="right")

    dynamic = [image, left_title, right_title]
    background, draw_order = _prepare_blitting(fig, ax, dynamic)

    try:
        for t in tqdm(times, disable=not show_progress, desc="Rendering frames"):
            image.set_data(snapshots[t])
            env = env_by_time.get(t, first_env)
            left_title.set_text(env.name)
            right_title.set_text(f"t = {t:.2f}")
            yield _blit_frame(fig, ax, draw_order, background)
    finally:
        plt.close(fig)


def _render_field_plot(field_names, time, env, values_by_field, cmap, vmin, vmax):
    """
    Render one static heatmap per field, side by side in one figure.

    Uses the same imshow/colorbar/title convention as `_render_field_frames`
    (one panel per field, colorbar labeled with the field's name, env name
    top-left / time top-right), but for a single point in time rather than
    a whole animation.

    Unlike `_display_and_save`, this does not call `plt.show()` itself: the
    figure is returned so the caller can add further annotations (e.g.
    `fig.axes[0].axvline(...)`) before displaying it — calling `plt.show()`
    early would, under Jupyter's inline backend, capture the figure
    immediately and miss anything added afterward.
    """
    width, height = env.size
    fig, axes = plt.subplots(
        1, len(field_names), figsize=(4.5 * len(field_names), 4), squeeze=False
    )

    for ax, field_name in zip(axes[0], field_names):
        image = ax.imshow(
            values_by_field[field_name],
            origin="lower",
            extent=[0, width, 0, height],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label=field_name)
        ax.set_title(env.name, loc="left")
        ax.set_title(f"t = {time:.2f}", loc="right")

    fig.tight_layout()
    return fig


def _even_rgb(frame):
    """
    An RGBA frame as RGB, padded to even dimensions.

    H.264 encodes in the yuv420p pixel format, which halves the chroma
    resolution in both directions and so needs an even width and height.
    """
    rgb = frame[..., :3]
    height, width = rgb.shape[:2]
    if height % 2 or width % 2:
        rgb = np.pad(rgb, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
    return rgb


def _write_video(frames, filepath, fps):
    """
    Encode frames as an H.264 video, one frame at a time.

    `macro_block_size=1` because the default (16) silently rescales any frame
    whose dimensions are not a multiple of 16 — a 642x482 figure comes out
    656x496 — whereas the encoder itself only needs the even dimensions
    `_even_rgb` guarantees.
    """
    with imageio.get_writer(
        filepath, fps=fps, codec="libx264", macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(_even_rgb(frame))


def _write_gif(frames, filepath, fps):
    """
    Encode frames as an animated GIF, one frame at a time.

    Pillow consumes `append_images` lazily, encoding each frame as it arrives,
    so handing it a generator keeps memory flat where building the full list
    of `Image`s first does not. Pillow remains the GIF encoder (rather than
    imageio, which writes the video formats here) because it is measurably
    the faster of the two at this.
    """
    images = (Image.fromarray(frame) for frame in frames)
    try:
        first = next(images)
    except StopIteration:
        raise ValueError("No frames to save.") from None

    first.save(
        filepath,
        save_all=True,
        append_images=images,
        duration=max(round(1000.0 / fps), 1),
        loop=0,
    )


def _save_frames(frames, save_path, filename, fps):
    """
    Save RGBA frames under `save_path`, as a video or an animated GIF
    according to `filename`'s extension.

    `frames` may be any iterable, including the generator `_render_frames`
    returns: both encoders consume it one frame at a time, so saving a long
    animation never holds more than a frame or two in memory.
    """
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)

    if os.path.splitext(filename)[1].lower() in _VIDEO_EXTENSIONS:
        _write_video(frames, filepath, fps)
    else:
        _write_gif(frames, filepath, fps)
    return filepath


def _display_and_save(frames, interval, save_path, filename, show=True):
    """
    Save (optionally) and interactively play back rendered RGBA frames.

    Shared by `Simulation.visualize_colony` and `Simulation.visualize_field`:
    once frames are being produced, showing (and optionally saving) them works
    identically regardless of what they depict.

    Interactive playback needs every frame available to scrub back and forth,
    so with `show=True` the frames are collected into a list first. With
    `show=False` they are instead streamed straight into the file as they are
    rendered and dropped immediately after, which keeps memory flat no matter
    how many frames there are; the saved path is returned in place of an
    animation.
    """
    fps = 1000.0 / interval

    if not show:
        if save_path is None:
            raise ValueError(
                "show=False with no save_path would render the animation and "
                "then discard it. Pass save_path to write it to a file, or "
                "show=True to display it."
            )
        return _save_frames(frames, save_path, filename, fps=fps)

    frames = list(frames)
    if save_path is not None:
        _save_frames(frames, save_path, filename, fps=fps)

    fig, ax = plt.subplots()
    ax.axis("off")
    fig.tight_layout(pad=0)
    image_artist = ax.imshow(frames[0])

    def update(frame_index):
        image_artist.set_data(frames[frame_index])
        return [image_artist]

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=True, repeat=True
    )
    plt.show()
    return anim


# Lineage-tree plot: marker per death cause (divided/alive cells get none).
_LINEAGE_DEATH_MARKERS = {
    "survival_condition": "x",
    "killed": "x",
    "out_of_bounds": ">",
}
_LINEAGE_BRANCH_COLOR = "0.2"
# Vertical gap (in leaf rows) between separate founders' trees.
_LINEAGE_ROOT_GAP = 1.0


def _lineage_layout(lineage, roots):
    """
    Assign each cell a y position: leaves get consecutive rows in DFS order
    (one tree after another, separated by `_LINEAGE_ROOT_GAP`), and every
    cell that divided sits at the mean y of its daughters. Iterative, so
    deep lineages don't hit the recursion limit.
    """
    y = {}
    next_row = 0.0
    for root in roots:
        stack = [(root, False)]
        while stack:
            cid, expanded = stack.pop()
            kids = lineage.children(cid)
            if not kids:
                y[cid] = next_row
                next_row += 1.0
            elif not expanded:
                stack.append((cid, True))
                stack.extend((k, False) for k in reversed(kids))
            else:
                y[cid] = float(np.mean([y[k] for k in kids]))
        next_row += _LINEAGE_ROOT_GAP
    return y


def _branch_segments(record, t_stop, y, times, values):
    """
    Split one cell's branch [birth, t_stop] at its recorded times, each
    piece colored by the value recorded at its start. Recorded rows outside
    the branch (e.g. a dead cell still listed afterwards) are ignored.
    """
    t0 = record.birth_time
    keep = (times >= t0) & (times < t_stop)
    times, values = times[keep], values[keep]
    if len(times) == 0 or times[0] > t0:
        # No sample at birth: extend the earliest known value back to it.
        first = values[0] if len(values) else np.nan
        times = np.concatenate([[t0], times])
        values = np.concatenate([[first], values])
    edges = np.append(times, t_stop)
    segments = [[(a, y), (b, y)] for a, b in zip(edges[:-1], edges[1:])]
    return segments, values


def plot_lineage_tree(
    lineage,
    history=None,
    color_by=None,
    root_id=None,
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    show_deaths=True,
):
    """
    Draw a lineage as a tree against time.

    Each cell is a horizontal branch from its birth to its division/death
    (or to `lineage.time` if still alive); a vertical connector joins the two
    daughters at each division. Leaves are stacked in depth-first order and
    each parent is centered on its daughters; separate founders are stacked
    with a gap. Deaths are marked at the branch tip ("x" for a survival
    condition or manual kill, ">" for leaving the environment).

    Args:
        lineage: a `Lineage` (e.g. `colony.lineage`).
        history: the Simulation history DataFrame (`sim.to_dataframe()`),
            required when `color_by` is given.
        color_by: a column of `history` (e.g. a species). Each branch is
            colored over time by that cell's recorded value.
        root_id: draw only this cell's subtree.
        ax: an existing Axes to draw on; a new figure is created otherwise.
        cmap, vmin, vmax: colormap and limits for `color_by` (limits default
            to the range of the plotted values).
        show_deaths: whether to mark deaths at branch tips.

    Returns:
        (fig, ax). Does not call `plt.show()`.
    """
    if color_by is not None:
        if history is None:
            raise ValueError("color_by requires the simulation history DataFrame.")
        if color_by not in history.columns:
            raise ValueError(f"'{color_by}' is not a column of the history.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    roots = [root_id] if root_id is not None else sorted(lineage.roots())
    ids = []
    for root in roots:
        ids.extend([root, *lineage.descendants(root)])
    if not ids:
        ax.set_xlabel("time")
        ax.set_yticks([])
        return fig, ax

    y = _lineage_layout(lineage, roots)

    grouped = {}
    if color_by is not None:
        sub = history.loc[history["cell_id"].isin(ids), ["cell_id", "time", color_by]]
        for cid, rows in sub.sort_values("time").groupby("cell_id"):
            grouped[cid] = (
                rows["time"].to_numpy(float),
                rows[color_by].to_numpy(float),
            )

    branch_segments, branch_values = [], []
    connectors = []
    deaths = {}
    for cid in ids:
        rec = lineage[cid]
        t_stop = rec.end_time if rec.end_time is not None else lineage.time
        if color_by is not None:
            times, values = grouped.get(cid, (np.empty(0), np.empty(0)))
            segs, vals = _branch_segments(rec, t_stop, y[cid], times, values)
            branch_segments.extend(segs)
            branch_values.extend(vals)
        else:
            branch_segments.append([(rec.birth_time, y[cid]), (t_stop, y[cid])])
        kids = lineage.children(cid)
        if kids:
            ys = [y[k] for k in kids]
            connectors.append([(rec.end_time, min(ys)), (rec.end_time, max(ys))])
        if show_deaths and rec.fate in _LINEAGE_DEATH_MARKERS:
            deaths.setdefault(rec.fate, []).append((t_stop, y[cid]))

    # Colored branches are many short per-step pieces; projecting caps
    # overlap them slightly so antialiasing doesn't leave hairline gaps.
    branches = LineCollection(
        branch_segments,
        linewidths=2.0,
        capstyle="projecting" if color_by is not None else "butt",
    )
    if color_by is not None:
        values = np.asarray(branch_values, dtype=float)
        finite = values[np.isfinite(values)]
        lo = vmin if vmin is not None else (finite.min() if finite.size else 0.0)
        hi = vmax if vmax is not None else (finite.max() if finite.size else 1.0)
        branches.set_array(np.ma.masked_invalid(values))
        branches.set_cmap(cmap)
        branches.set_clim(lo, hi)
        fig.colorbar(branches, ax=ax, label=color_by)
    else:
        branches.set_color(_LINEAGE_BRANCH_COLOR)
    ax.add_collection(branches)
    ax.add_collection(
        LineCollection(connectors, colors=_LINEAGE_BRANCH_COLOR, linewidths=1.0)
    )
    for fate, points in deaths.items():
        xs, ys = zip(*points)
        ax.scatter(
            xs,
            ys,
            marker=_LINEAGE_DEATH_MARKERS[fate],
            color="crimson",
            zorder=3,
            label=fate.replace("_", " "),
        )
    if deaths:
        ax.legend(loc="best", fontsize="small")

    ax.autoscale_view()
    ax.set_xlabel("time")
    ax.set_ylabel("cell")
    leaf_ids = [cid for cid in ids if not lineage.children(cid)]
    if len(leaf_ids) <= 40:
        ax.set_yticks([y[c] for c in leaf_ids], [str(c) for c in leaf_ids])
    else:
        ax.set_yticks([])
    ax.invert_yaxis()
    return fig, ax
