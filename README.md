# Phase Mask Quantizer

Turn a **continuous phase profile** (`.mat`, `.npy`, or a grayscale `.png` / `.bmp`) into a **discrete,
fabrication-ready height map**, and export it as PNG level maps and/or a watertight stepped **STL**
(for two-photon polymerization, grayscale lithography, etc.).

Single script with a GUI and a command-line mode: [`phase_mask_quantizer.py`](phase_mask_quantizer.py).

## Install

Python 3.10+ with tkinter (included in the python.org and Anaconda installers).

```bash
pip install -r requirements.txt
```

## Run

**GUI** (double-click launchers)

| OS | How |
|----|-----|
| macOS | run `launchers/build_mac_app.sh` once, then double-click **Phase Mask Quantizer.app** |
| Windows | double-click `launchers/Phase Mask Quantizer.bat` |
| any | `python phase_mask_quantizer.py` |

The GUI shows the input phase (grayscale, wrapped 0–2π) next to the quantized level map, the physical
footprint, step / max height, refractive indices, RMS quantization error and efficiency. Every
setting is a dropdown you can also type into. **Export** (or Cmd/Ctrl+E) writes the files.

**Command line**

```bash
python phase_mask_quantizer.py INPUT PIXEL_SIZE_UM WAVELENGTH_NM [png|stl|both] [--step-nm 200 | --levels 6]
python phase_mask_quantizer.py --help      # all options
```

## How it works

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

## Outputs

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

## Try it

`examples/` has ready-to-use 256×256 phase images (black = 0, white = 2π): `lens_phase.png`,
`vortex_phase.png`, `smooth_phase.png`. Regenerate them with `python examples/make_examples.py`.

```bash
python phase_mask_quantizer.py examples/lens_phase.png 1.0 532 both --step-nm 200 --output-dir out
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers both quantization modes (including the PhlatCam formula), image / `.npy` / `.mat` loading, STL
closure and volume, output names, footprint and export files. CI runs them on every push.

## Library use

```python
import phase_mask_quantizer as pq
report = pq.run("mask.mat", pixel_pitch_um=1.0, wavelength_nm=532, output="both", step_nm=200)
```

## License

[MIT](LICENSE)
