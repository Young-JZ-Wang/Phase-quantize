# NanoSlicer

One app for the whole chain: **continuous phase profile → discrete height map (PNG / STL) → sliced fields →
Nanoscribe job files**, for two-photon polymerization and grayscale lithography.

| Tab | What it does |
|-----|--------------|
| **Quantizer** | Continuous phase (`.mat`, `.npy`, `.png`, `.bmp`) → N-level height map, exported as PNG and/or a watertight stepped STL. The quantizer of the earlier *Phase Mask Quantizer*, with one added button, **Send to Job File Generator**; its behaviour and outputs are unchanged. |
| **Job File Generator** | Height map (`.png` `.bmp` `.mat` `.npy`) or STL → `job.gwl` + `data.gwl` + one file per field, with **standard** or **adaptive** slicing and the writing parameters from a **recipe**. Also converts a height map to STL. |

## Install

Python 3.10+ with tkinter (included in the python.org and Anaconda installers).

```bash
pip install -r requirements.txt
```

The Quantizer tab needs only numpy, pillow, scipy (+ h5py for MATLAB v7.3 files). The Job File Generator also
needs `opencv-python-headless`, `tables` and `matplotlib` (TipSlicer), `scikit-image` (height-gradient-guided segmentation) and
`numpy-stl` for an STL input.

## Run

| OS | How |
|----|-----|
| macOS | run `launchers/build_mac_app.sh` once, then double-click **NanoSlicer.app** |
| Windows | double-click `launchers/NanoSlicer.bat` |
| any | `python nanoslicer.py` |

`python quantizer.py INPUT PIXEL_SIZE_UM WAVELENGTH_NM ...` (or `python nanoslicer.py quantize ...`) is the quantizer command line.

## Two ways to work

```
Quantizer only                         Quantizer → Job File Generator            Job File Generator on its own
load PNG / MAT / NPY                   load + quantize in the Quantizer tab      open an STL or a height map in tab 2
quantize, preview                      "Send to Job File Generator →"            (Source → From file)
export PNG / STL                       choose slicing method + recipe
                                       look at the slicer preview, generate
```

The Quantizer tab loads, quantizes, previews and exports (PNG / STL). The Job File Generator tab handles the
recipe, the slicing method, the slicer preview and the job files. The two share data only through one object
(`state.AppState` holding a `state.Design`), never by copying values between fields.

### Sending a design from the Quantizer

**Send to Job File Generator →** (above *Export*) publishes the current quantized result and opens tab 2. The
Job tab then shows *Source: From Quantizer* and keeps:

* the source file path and the **quantized height map** itself (used directly, no round trip through a PNG),
* **pixel size**, **number of levels**, **height per level** and peak height, and the **physical dimensions**,
* the **PNG / STL** written for that result: exporting after sending attaches their paths to the design; they are
  ignored if you change the quantization afterwards (they no longer belong to the design). *Height map → STL…* in
  tab 2 builds an STL straight from the quantized heights and attaches it too.

The pixel size must be set in the Quantizer before sending. The design fields in tab 2 only mirror the state; change
them in the Quantizer and send again.

### Slicer preview

The centre of the Job File Generator tab shows **how the selected method divides the structure**, before anything is
written: field borders (dashed, semi-transparent red, so the phase / height image stays readable), overlap bands
(orange), field centres (yellow) and the writing order (cyan, first field green, last magenta). The **Show** toggle in
the preview header switches it on and off; while it is off nothing is computed. It is the plan the job uses (same code, `job_generator.height_map_layout`):

* **Standard**: the rectangular field grid of the recipe's *field size*, row by row (instant).
* **Adaptive**: TipSlicer's edge-voting layout, Voronoi cells in writing order. A fast first look appears, then a
  full-resolution pass replaces it (*refining…*); for a design of up to 1100 px per side it is the exact layout.
* It updates by itself when you change the **input file**, the **slicing method**, the **field size**, the **field
  radius**, the **overlap**, the base thickness, the image orientation, the design values, or the recipe.
* An **STL** input is previewed from its projection onto the xy plane (objects taller than 50 µm are also split in z
  by the job).

## Job File Generator

1. **Source**: *From Quantizer* (the design you sent) or *From file*: a height map (`.png .bmp .mat .npy`) or an STL.
   For a file, fill in pixel size, height max (the peak height; images need it, `.mat` / `.npy` may leave it empty to use
   their own values in the chosen unit) and levels (snap to N equidistant levels, empty = keep the heights).
   Base thickness applies to both.
2. **Slicing method**
   * **Standard**: rectangular fields of the recipe's field size (100 µm), written row by row, the way DeScribe splits a DOE.
     For an STL input it is TipSlicer's regular square tiling.
   * **Adaptive**: TipSlicer's adaptive stitching. Field centres are chosen by edge voting so that stitching seams
     avoid the features; fields are Voronoi cells written in a travelling-salesman order (Concorde if `$CONCORDE` or `PATH`
     points to it, otherwise nearest neighbour). It pays off for patterns with features to protect; on a design that
     is a full solid square it behaves like a loosely placed grid.
3. **Recipe**: speed, power, slicing and hatching distance, shells, hatch angle, field size / radius, scan field,
   overlap, FindInterfaceAt, piezo settling time, galvo acceleration, stage velocity. The default is **`63x - DIP`**
   (built in: it can be selected and edited, never overwritten or deleted). *Save as…* stores your edits under a new
   name, *Delete* removes a saved recipe, *Reset* reloads the selected one. Saved recipes live in
   `~/.nanoslicer/recipes.json` (override the folder with `NANOSLICER_HOME`).
4. **Generate job files** (Cmd/Ctrl+G). Progress, a log and *Cancel* are shown while it runs.
5. **Height map → STL…** writes a closed solid of flat-topped pixel columns (the steps stay sharp; equal-height
   neighbours are merged unless you turn that off under *Advanced*).

### Height-based adaptive slicing

Ported from TipSlicer's *Adaptive Stitching with Nanoscribe* notebook (microlens arrays on a thick base). Tick
**Height-based segmentation** under *Slicing method* (adaptive only) and give the height *z* (µm, base included; it
defaults to the base thickness). It is off by default and then nothing changes. Three things happen:

1. **Segmentation at a height**: the field centres are chosen from the cross-section of the structure at *z* (the
   features above the base), not from the whole footprint, so the seams avoid the features. The plain base the discs
   do not reach is then covered by the usual edge voting, so every part of the footprint is still written. This is what
   makes adaptive slicing useful for a DOE or lens array on a full-square base.
2. **Coarse slicing below *z***: the recipe's *Base slicing* distance replaces the slicing distance below *z*
   (0 = same), as in the notebook's `zzbase` / `zzlenses`.
3. **More power below *z***: the recipe's *Base power ×* multiplies the laser power below *z* (`MultLaserPower` in
   the field files; the notebook's `intensityMultiplier`).

The preview, the job and the exported adaptive PNG all use the same segmentation. From the command line:
`--segment-height 0.2 --set base_slicing_distance=2 --set base_power_multiplier=5`. For an STL input the cross-section is
taken from the low-resolution stack at *z*.

### Height-gradient-guided segmentation

The third way to find the fields of the adaptive method (*Slicing method → Adaptive → Segmentation*). The other two see
only a silhouette: the footprint, or the cross-section at one height (a single 0/1 mask, whose outline is all they can
use). This one reads the **whole quantized height map**, so every transition (level 1→2, 2→3, …) can attract a seam:

1. `gradH = |∇H|` is taken on the actual height values and normalised to [0, 1] (99th percentile of the non-zero gradients).
2. Writing centres cover the aperture within the write radius: a hexagonal lattice whose phase puts the seeds in smooth
   places, each seed nudged to the smoothest pixel nearby, redundant seeds removed. Odd-shaped apertures are completed by
   a greedy disc cover, so every pixel is always within the write radius of a centre.
3. Every pixel goes to the centre it reaches at the lowest travel cost, where a pixel costs `1 + ridge_cost · gradH`
   (default 25). The floods stop at the steps, so block borders settle on them; in smooth areas the cost is uniform and
   the borders are the usual nearest-centre ones. A centre only competes inside its own square writing frame, so a block can
   never leave its frame, and no part of the aperture is left without a block.

The heights and the optical design are not touched; only the block masks change. The masks plug into the same slicer as
the other modes. The job also writes `<name>_segmentation_labels.png` (16-bit block ids, id + 1, 0 = outside) and
`<name>_sliced_comparison.png` (standard square | threshold-based adaptive | height-gradient-guided, side by side).

```bash
python nanoslicer.py job mask.png out/ --method adaptive --segmentation height_gradient_guided --pixel-size 2 --height-max 1 --levels 6
python nanoslicer.py compare mask.png comparison.png --pixel-size 2 --height-max 1 --levels 6 --labels labels.png
python examples/demo_gradient_segmentation.py        # the demo, on examples/demo_height_map.png
```

How much it helps depends on the design. On a map with few steps in a lot of smooth area (a sparse DOE) it moved the
share of block border lying on a height step from 38% (nearest centre) to 56%. On `examples/demo_height_map.png`, dense
stripes where 39% of the pixels are on a step, nearly every border is on a step by chance (standard 70%, threshold 69%),
and the gain is small (76%). It is a STL-free mode: it needs a height map as the input.

### Phase-quality-guided segmentation

The fourth way to find the fields of the adaptive method, for phase elements (`phase_quality.py`; the module docstring
explains the reasoning). What matters at a stitching seam in a phase mask is **phase continuity**: a seam through a smooth
phase area splits one optical feature and puts a small field misregistration where the design had none, while a seam
along an existing height step hides in a discontinuity the structure already has.

How it extends the two adaptive-stitching papers:

* *Adaptive stitching* (Dehaeck, Scheid, Lambert 2018) picks the writing centres by edge voting on a **binary** mask and
  takes the Voronoi cells as the fields: it sees one silhouette and puts the borders wherever Voronoi does.
* *Miniscope3D* (2020) gives every point to the microlens whose local **height function** is highest, so the seams fall on
  the lens boundaries. A wrapped, quantized DOE has no local height functions to compare.
* Here the structure is the internal step pattern of one height map, so the seams are placed by the **full height-gradient
  map**: `gradH = |∇H|` (normalised to [0, 1]) and a seam cost `eps + (1 − eps)·exp(−k·gradH)`, cheap on a step, dear in
  smooth areas.

The fields are the minimum-cost partition under the printer's rules: every aperture pixel in exactly one block; a pixel
only in a block whose square writing frame holds it; every block one connected piece. It starts from the
height-gradient-guided blocks and re-optimises every border between two neighbouring blocks as a **graph cut** (a
minimum s-t cut, exact, via scipy's max-flow; a few sweeps, never raising the cost), then merges stray pieces into the
neighbour they touch. Blocks that end up empty are dropped. The writing order is the travelling-salesman order (Concorde
if available) improved by 2-opt, because the stage travel between fields is what the order is for; the standard method
keeps DeScribe's row-by-row order.

**Writing order and the optional exact solver** (`route_solvers.py`). The order of the fields is a travelling-salesman path
(open: it starts and ends anywhere). It applies to every adaptive mode (footprint edges, height threshold, height-gradient-guided,
phase-quality-guided); a standard-method height map keeps DeScribe's row order, an STL input (either method) is ordered by the same solver. The route is always improved by 2-opt, which needs nothing extra. If
[pyconcorde](https://github.com/jvkersch/pyconcorde) is installed (optional; it is not in `requirements.txt`, and building it
needs a C compiler), the job first asks Concorde for the exact optimal open path (solved as a tour with one dummy node of
distance 0, cut at the dummy; distances rounded to 0.01 µm; at most 1500 fields and 60 s by default). Only a route
Concorde proves optimal is called optimal and labelled `pyconcorde`. If it is missing, fails, hits the time limit,
returns an invalid tour or the instance is unsuitable, the 2-opt route is used and labelled `2-opt`, with the reason
logged. The segmentation, block masks and seam metrics never depend on the solver; only the order (and so the travel)
can differ. The solver and whether it was optimal are reported in the job report and in the job file's notes, in the
`compare` table and its `--metrics` JSON (`route_solver`, `route_optimal`). The live preview always uses 2-opt so it never
waits for an exact solve. Command line: `--route-solver auto|2-opt`, `--route-time-limit SECONDS`. Other backends can be
added with `route_solvers.register_solver`.

The job writes the same files as the other label-image modes (`<name>_segmentation_labels.png`,
`<name>_sliced_comparison.png`, `<name>_sliced_adaptive.png`, `<name>_sliced_standard_square.png`) plus
`<name>_writing_path.png`; the comparison then shows all five modes with a metrics table.

```bash
python nanoslicer.py job mask.png out/ --method adaptive --segmentation phase_quality_guided --pixel-size 2 --height-max 1 --levels 6
python nanoslicer.py compare mask.png cmp.png --pixel-size 2 --height-max 1 --levels 6 --metrics metrics.json --path-image path.png
```

**Metrics** (`slicer_preview.layout_metrics`): the same grid and the same seam cost for every mode, so they compare. Blocks;
total seam length; mean seam cost; share of the seam lying on a strong height step; disconnected blocks, extra pieces and
tiny islands; stage travel along the writing order. On `examples/demo_height_map.png` (dense stripes):

| mode | blocks | seam (mm) | mean seam cost | seam on a step | islands | travel (mm) |
|---|---|---|---|---|---|---|
| standard square | 441 | 81.9 | 0.771 | 9% | 0 | 82.0 |
| adaptive, footprint edges | 431 | 97.3 | 0.771 | 10% | 0 | 44.7 |
| adaptive, threshold | 476 | 103.9 | 0.750 | 15% | 0 | 50.2 |
| adaptive, height-gradient-guided | 449 | 103.2 | 0.631 | 29% | 1 | 49.8 |
| adaptive, phase-quality-guided | 435 | 87.1 | 0.547 | 35% | 0 | 46.0 |

### Output (named like a DeScribe DOE package)

With the **adaptive** method the job also writes two images of the same structure, divided both ways, to compare the
layouts: `<name>_sliced_standard_square.png` (rectangular fields) and `<name>_sliced_adaptive.png` (the field centres and
writing order the job used). They are drawn on a fine grid (at least 1000 px) in the same style as the preview.

| File | Content |
|------|---------|
| `<name>_job.gwl` | writing configuration, `$solidLaserPower`, `$solidScanSpeed`, `$interfacePos`, includes the data file |
| `<name>_data.gwl` | stage moves, `% FIELD i\|j`, `FindInterfaceAt`, one include per field |
| `<name>_files/<name>_<i>_<j>.gwl` | one field each, in galvo coordinates relative to the field centre (adaptive: `<name>_<number>.gwl`) |
| `<name>.png`, `<name>.recipe`, `<name>_job.recipe` | model image and DeScribe-style recipes (height maps with levels) |
| `work/` | H5 stacks and TSP files, only with *Keep intermediate files* |

**Block shear angle** (recipe fields `block_shear_angle` and `block_shear_direction`, default 0 = unchanged). In DeScribe the
blocks are not cuboids but prisms whose faces lean by the *Block shear angle* from the substrate normal, so the objective's
light cone is not shadowed by the neighbouring block that is already printed ([NanoGuide: Block Splitting and
Stitching](https://support.nanoscribe.com/processes/print-strategies/block-splitting-and-stitching/), login required; DeScribe
recommends roughly 15-30 deg for its block stitching). Here the seams between the fields lean the same way: a point at height
`z` above the block base belongs to the field that contains `(x, y) - z * tan(angle) * (cos d, sin d)`, with `d` the direction
(0 = +x, 90 = +y). Every field moves the same way, so the fields still tile the structure without a gap or an overlap; the
field centres, the segmentation and the writing order do not change, and each field's stack frame grows by the largest shift
(a warning is logged if it no longer fits the scan field). Use the recipe panel, `--block-shear-angle DEG
[--block-shear-direction DEG]` or `--set block_shear_angle=DEG`; works for height maps and STL input. Not written to the
`.recipe` (no documented key for DOE recipes) but noted in the job file. For a 1 um DOE the shift is well under 1 um, so
this matters for tall STL structures; DeScribe's own DOE import does not use it.

**Global interfaceFound.** By default `FindInterfaceAt $interfacePos` (interfaceFound) runs before every field. With
`--global-interface-found` (CLI) or the *Global interfaceFound* checkbox (UI), `JobSettings.global_interface_found`, it
runs once, at the first field, and that interface position is reused for all later fields; the later calls become
comments. Segmentation, field order, stage moves and the field files are unchanged. (DeScribe itself calls it once only when the structure also exceeds the block in z; for a structure that exceeds the block only in x and y, as height maps do, it calls it before every block, which also corrects a tilted substrate — see [NanoGuide](https://support.nanoscribe.com/processes/print-strategies/block-splitting-and-stitching/).) For an existing data file:
`python nanoslicer.py interface <name>_data.gwl [-o out.gwl] [--remove] [--command FindInterfaceAt]`.

Compared with a DeScribe DOE package of the same design, the job and data files are line-for-line identical in
standard mode. The field files are text `.gwl` (DeScribe writes binary `.gwlb`) with about 13% fewer points. **Check a
job in DeScribe / Nanowrite before printing.**

### Command line

```bash
python nanoslicer.py job mask.png out/ --pixel-size 2 --height-max 1 --levels 6 --method adaptive --set laser_power=60
python nanoslicer.py stl mask.npy 2 --height-max 1 --levels 6 -o mask.stl
python nanoslicer.py preview mask.png preview.png --pixel-size 2 --height-max 1 --levels 6 --method adaptive
python nanoslicer.py recipes list            # also: show NAME, delete NAME
python nanoslicer.py quantize INPUT PITCH WAVELENGTH ...   # the quantizer command line
```

## Quantizer tab

Turn a **continuous phase profile** (`.mat`, `.npy`, or a grayscale `.png` / `.bmp`) into a **discrete,
fabrication-ready height map**, and export it as PNG level maps and/or a watertight stepped **STL**
(for two-photon polymerization, grayscale lithography, etc.). The tab shows the input phase next to the quantized
level map, the physical footprint, step / max height, refractive indices, RMS quantization error and efficiency.
Every setting is a dropdown you can also type into. **Export** (or Cmd/Ctrl+E) writes the files.

### How it works

1. Load the phase and wrap it to [0, 2π). Images map black → 0, white → 2π (8- or 16-bit).
2. Height for a full 2π: `λ / (n_material − n_medium)`.
3. Quantize, in one of two modes:
   - **Levels**: `N` equal steps, `step = λ / (N·Δn)`, nearest level.
   - **Step height** (`--step-nm`; the GUI default, 200 nm): fixed step `dh` as in PhlatCam,
     `hq = dh · floor(h / dh)` (or `round`); the number of levels follows from the 2π depth.
   `--phase-scale 2` ("Phase multiplier" in the GUI) doubles the phase first, as PhlatCam's MATLAB
   script does (`mod(2*phMm, 2*pi)`). The GUI defaults to PhlatCam's 200 nm step.
4. **Resampling** (optional, before quantizing): `Original` (default, same pixel grid), `Up` or `Down` by a
   factor (presets 1.5, 2, 4, 8, or any number; `--resample up --factor 2`). The factor applies per
   axis, so `Up 2×` gives 4× the pixels. The pixel size you enter is not changed, so the
   physical footprint is the **output** image size × pixel size (it grows with `Up`, shrinks with `Down`). The unit phasor `exp(iφ)` is interpolated, so 2π wraps stay clean.
5. Export, in two stages (below).

Material presets: `air`, `fused_silica`, `su8`, `pdms`, `ip_dip`; or type any refractive index.
The substrate index is reported but does not change the height.

### Outputs

Written next to the input (or to the chosen folder), in two stages: first the continuous phase, then the
discrete design on the resampled grid (resampled names end in `_up2x` / `_down4x`):

| File | Content |
|------|---------|
| `<name>_continuous.npy` / `_continuous.png` | **Stage 1, original continuous phase**, before any resampling or discretization: exact float64 radians, and a 16-bit wrapped image (black = 0, white = 2π). Turn off with `--no-continuous` or the GUI checkbox. |
| `<name>.png` | **8-bit** quantized height map, divided by its own maximum (same as PhlatCam's `imwrite(phMmHq/max(phMmHq))`); only the discrete levels appear |
| `*_maskN_depth…nm.png` | optional binary litho masks (levels = 2^m) |
| `*.stl` | watertight stepped solid with base slab (units selectable) |
| `*_report.json` | settings, heights, footprint, RMS error, level fill |
| `*_levels.npy`, `*_height_m.npy` | optional arrays |

File name: `phHeight_<input>_<maxH>umMaxH_<pitch>um_<size>um_q<step>nm_lam<nm>`, where
`size = max(Nx, Ny) × pixel size` of the output image (the physical footprint).

### Try it

`examples/` has ready-to-use 256×256 phase images (black = 0, white = 2π): `lens_phase.png`,
`vortex_phase.png`, `smooth_phase.png`. Regenerate them with `python examples/make_examples.py`.

```bash
python quantizer.py examples/lens_phase.png 1.0 532 both --step-nm 200 --output-dir out
```

### Library use

```python
import quantizer as pq
report = pq.run("mask.mat", pixel_pitch_um=1.0, wavelength_nm=532, output="both", step_nm=200)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the quantizer (both quantization modes including the PhlatCam formula, image / `.npy` / `.mat` loading, STL
closure and volume, output names, footprint and export files), the height-map → STL converter, recipes, the job
generator (both slicing methods, job / data / field files, geometry fidelity, cancel, errors, heights handed over from
the Quantizer), the shared state, the slicer preview (layouts, rendering, STL footprints) and the GUI workflows
(send from the Quantizer, preview updates, standalone STL, generation; skipped without a display). CI runs them on every push.

## License

[MIT](LICENSE) for this repository, **except** `third_party/tipslicer/`: it contains
[TipSlicer](third_party/tipslicer/NOTICE.md) by Sam Dehaeck, which is licensed under the **GNU GPL v3**
(`third_party/tipslicer/LICENSE`). It is imported only by `job_generator.py`. If you distribute the app together
with that folder, the GPL applies to the combined work; remove the folder (and the Job File Generator) to keep a
pure MIT distribution.
