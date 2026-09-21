# Visualization speed-up plan

Working branch: `visualizationSpeedup`

## Context
`Simulation.visualize_colony` / `visualize_field` cost 10–100x more than the
simulation that feeds them, which is the limiting factor on large-population
runs. The aim is to cut that cost without changing what the animations look
like. Each stage below is benchmarked before and after, and the rendered output
is checked against the previous implementation pixel for pixel.

## Status

| Step | State | Result |
|------|-------|--------|
| 1. Benchmark harness | **done** | `examples/bench_visualization.py`; baseline captured below |
| 2. Tier 1 — one collection, vectorized geometry and colors | **done** | render 3.7x faster at 573 cells, 6.4x at 1125 cells, output bit-identical |
| 3. Tier 2 — blit the static background | **done** | render 8.2x faster again; 30x vs. baseline at 573 cells, 41x at 1125 cells, output bit-identical |
| 4. Tier 3 — stream frames to the writer, MP4 output | **done** | MP4 ~2.8x faster to write and ~10x smaller than GIF; streaming keeps memory flat; GIF output byte-identical |
| 5. Tier 4 — parallel frame rendering | not started | |
| 6. Tier 5 — rasterize cells without matplotlib | not started | only if still needed |
| 7. Tier 6 — columnar history instead of per-cell dicts | not started | only if still needed |

Nothing is committed; all changes are in the working tree. Step 4 is the first
step that changed the public API (a new `show` argument and the `.mp4` option),
so `README.md` was updated to cover the GIF-vs-MP4 choice; steps 1–3 were
internal and needed no user-facing documentation.

### Measurements

`python examples/bench_visualization.py --t-max 9 --save` (451 frames, 573 cells
in the final frame):

| Stage | Baseline | After step 2 | After step 3 | After step 4 |
|-------|----------|--------------|--------------|--------------|
| simulate | 5.49s | 6.49s | 5.71s | 5.43s |
| to_dataframe | 0.08s | 0.08s | 0.07s | 0.06s |
| render | **62.41s** | **17.01s** | **2.08s** | 1.84s |
| save_gif (1.05 MB) | 4.80s | 6.64s | 4.76s | 4.76s |
| save_mp4 (0.11 MB) | — | — | — | **1.53s** |
| render+encode streamed to mp4 | — | — | — | 3.50s |
| per rendered frame | 138.4 ms | 37.7 ms | 4.6 ms | 4.1 ms |
| visualization vs. simulation | 12.3x | 3.7x | 1.2x | 1.2x (mp4) |

A second, larger case (501 frames, 1125 cells): render **133.6s → 20.9s → 3.3s**,
**41x** overall. With a field overlay (`--field`, 573 cells) rendering is 3.97s,
8.8 ms/frame — the overlay roughly doubles per-frame cost, since a second image
has to be recomposited every frame.

Step 2's gain grew with cell count (it removed per-cell work); step 3's is
roughly constant per frame (it removed per-frame work), so together they cover
both regimes. `simulate` and `save_gif` vary a little run to run; neither was
touched.

After step 4, writing an MP4 (1.53s, 0.11 MB) costs about a third of what
writing the equivalent GIF does (4.76s, 1.05 MB) and produces a file roughly 10x
smaller. GIF encoding is unchanged — see below for why it was left alone.

Peak memory for a streamed render-and-encode is **63 MB against the 554 MB** the
frame list needs. The 63 MB no longer scales with frame count; what remains is
dominated by the precomputed outline array, which scales with total recorded
rows (49k rows x 24 vertices ≈ 19 MB here).

### What step 2 changed (`src/multicellular/utils/visualization.py`)
- `_capsule_outlines` computes every cell's outline for the whole run in one set
  of array operations, from a cap template built once per render. It takes the
  rotation straight from the orientation vector, with no `arctan2`/`cos`/`sin`
  per cell. `_capsule_outline` is kept as a single-cell wrapper.
- `_channel_column` replaces `_channel_value`: one RGB channel for every row at
  once. A species absent from a row (NaN, from a colony whose cells carry
  different networks) now reads as zero concentration instead of producing a
  NaN color.
- `_render_frames` draws all of a frame's cells as one `PolyCollection`, reusing
  a single artist across frames via `set_verts` / `set_facecolor`, instead of
  creating, adding and removing one `Polygon` patch per cell per frame. Frames
  are located by `searchsorted` over the time column rather than by `groupby`
  plus `iterrows`.
- `n_points` (vertices per hemispherical cap) is now a parameter, defaulting to
  the previous 12 so output is unchanged.

Two matplotlib details were needed to keep the output pixel-identical, both
commented at the call site:
- A `Collection` defaults to **round** joins, a `Polygon` patch to **mitered**.
- Given a single path and one of everything else, `Collection.draw` switches to
  a marker-stamping fast path (`draw_markers`) that places the stamp up to a
  pixel away from where `draw_path` would. Passing two identical `antialiaseds`
  flags fails that all-length-one test, so one-cell frames render the same way
  as every other frame.

### What step 3 changed (`src/multicellular/utils/visualization.py`)
Two new helpers, `_prepare_blitting` and `_blit_frame`, shared by
`_render_frames` and `_render_field_frames`. The figure is drawn once, the
result cached with `copy_from_bbox`, and each frame then restores that cache and
redraws only what changed — the cells, the field overlay and the two titles.
The titles are now `Text` artists whose text is set per frame, instead of
calling `ax.set_title` per frame. If the backend cannot blit (pdf/svg have no
pixel buffer), it falls back to a full redraw per frame.

Three matplotlib behaviors had to be handled to keep the output identical, all
commented where they are dealt with:
- A full redraw paints the **spines after** the axes contents, so a changing
  image covers the inner half of the spine line. The spines are therefore
  redrawn on top of each frame.
- **`animated` does not exclude an `AxesImage`** from a draw: `Axes.draw` keeps
  animated images in its draw list on purpose (`axes/_base.py:3043`). An
  animated semi-transparent field overlay was blended into the background *and*
  again into every frame, coming out at twice strength. Images are hidden for
  the background draw instead.
- **A hidden title gets misplaced**: `Axes.draw` recomputes title positions from
  what is visible, and an invisible title is pushed above the top of the figure.
  So non-image artists are marked `animated` rather than hidden.

The net of those three: `animated` for everything except images, `visible=False`
for images, and spines redrawn last.

### Verification done so far
- 186/186 existing tests pass (`python -m pytest tests/ -q`).
- Old (pre-Tier-1) vs. current renderers compared frame by frame across six
  configurations: colony with and without a field overlay, RGB channels driven
  by a species column, walls and an out-of-bounds patch, `stride=3`, dead cells
  scattered through the run, a colony of exactly one cell, an environment whose
  name changes mid-run, and `_render_field_frames` over a diffusing field.
  **0 differing pixels** in every case.
- The no-blitting fallback path was exercised separately and also matches the
  old output exactly.
- `_capsule_outlines` vs. the original per-cell `_capsule_outline` over 200
  random cells: max coordinate error 8.9e-16. Zero-length orientation handled.
- `_channel_column` matches the old `_channel_value` row by row for a None
  species, a normal scale, a zero scale and a missing column.
- After step 4: rendered frames still pixel-identical; the written GIF is
  byte-identical to the old encoder's; the MP4 reads back with the right frame
  count and unscaled dimensions; streaming peaks at 9.9 MB against 176.7 MB for
  the same animation collected into a list; `show=False` returns a path while
  the default still returns a `FuncAnimation`; `show=False` without `save_path`
  raises `ValueError`.

### What step 4 changed
- `_render_frames` and `_render_field_frames` are now **generators**: they yield
  each frame instead of returning a list, so a caller writing to a file can
  encode each one and drop it. Both wrap their loop in `try/finally` so the
  figure is closed even if the caller abandons the generator.
- `_save_frames` dispatches on the filename extension — `.mp4` goes to
  `_write_video`, anything else to `_write_gif` — and both encoders consume the
  frames one at a time. `imageio`/`imageio-ffmpeg` were already installed and
  bundle their own ffmpeg, so MP4 needs no new dependency and no system ffmpeg.
- `_display_and_save` takes `show`. With `show=True` it collects the frames (as
  before) because interactive playback needs them all; with `show=False` it
  streams them to the file and returns the path. `show=False` with no
  `save_path` raises a `ValueError` rather than rendering and discarding.
- `Simulation.visualize_colony` / `visualize_field` take `show=True`, and their
  docstrings cover the format choice.

Two decisions worth recording, both made from measurements rather than
assumption:
- **GIF stays on Pillow.** The plan was to move it to `imageio`, but imageio's
  GIF writer measured *slower* (2.61s vs. 1.28s on 120 frames), so the switch
  would have been a regression. Pillow consumes `append_images` lazily, so it
  streams just as well; the GIF path gained the memory win without changing its
  output, which is **byte-identical** to before.
- **`macro_block_size=1` for video.** imageio-ffmpeg's default of 16 silently
  rescales any frame whose dimensions are not a multiple of 16 — a 642x482
  figure comes out 656x496. Frames are instead padded to even dimensions
  (`_even_rgb`), which is all H.264's yuv420p actually requires.

## Remaining work

### Tier 4: parallel frame rendering
Frames are independent. Split `times` into chunks across `joblib.Parallel`
workers, as `src/multicellular/utils/parallel.py` already does for replicates,
each worker building its own Agg figure. Pass pre-sliced NumPy arrays, not the
DataFrame. Note that rendering is now only 2.08s of a 12.62s benchmark, so
parallelism has little left to win here — reconsider it only for runs with far
more frames than these, and after Tier 3, since streaming to a single writer
constrains how frames can be farmed out and reassembled in order.

### Tier 5 (only if needed, for very large colonies)
Render the static background once with matplotlib, then fill cell outlines
directly into a copy of it with `cv2.fillPoly` (opencv is installed;
`LINE_AA` antialiases), mapping data to pixel coordinates via the axes
transform. Handles 100k+ polygons per frame, at some cost in code complexity
and styling fidelity. Tier 2 already caches the static background, so this
would be a change to how the cells alone are drawn. Only worth it if colonies
in the tens of thousands of cells prove too slow — measure with
`--t-max 14`-ish first.

### Tier 6 (only if needed)
`Simulation.record` builds one Python dict per cell per step and
`to_dataframe` turns the list into a DataFrame. Recording one array per step
instead would speed up `run` as well as the renderer. Currently only 0.08s of
the benchmark, so this is not yet worth doing.

## Verification for the remaining steps
- Re-run `python examples/bench_visualization.py --t-max 9 --save` (and
  `--field`, and a larger `--t-max`) before and after each step; record the
  per-stage timings in the table above.
- Keep comparing rendered frames against the pre-Tier-1 renderer
  (`git show 7d2367b:src/multicellular/utils/visualization.py`), which is still
  the reference for what the animations should look like. Tiers 1 and 2 held at
  zero differing pixels; Tiers 3 and 5 legitimately change the output (different
  encoder, different rasterizer), so those need a tolerance or an eyeball check
  instead.
- `python -m pytest tests/ -q` stays at 186 passed.
- Run the notebooks that call these functions end to end, including the ones
  that save: `examples/feature_tour.ipynb`, `danino_oscillator_demo.ipynb`,
  `quorum_sensing_demo.ipynb`, `mother_machine_demo.ipynb`.
