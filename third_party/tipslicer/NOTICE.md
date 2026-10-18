# TipSlicer (bundled)

`TipSlicer.py` is **TipSlicer** by Sam Dehaeck, an alternative slicer for
Nanoscribe GT systems with the adaptive stitching of

> S. Dehaeck, B. Scheid & P. Lambert, *Adaptive stitching for meso-scale printing with two-photon lithography*,
> Additive Manufacturing 21 (2018) 589-597.

It is licensed under the **GNU General Public License v3** (see `LICENSE` in this folder), unlike the rest of this
repository (MIT). It is kept separate on purpose; it is only imported by `job_generator.py`.

## Changes made for NanoSlicer

Everything else is the original code.

* NumPy 2 / OpenCV 4 compatibility: `np.int` -> `int`, `np.float` -> `float`, `np.row_stack` -> `np.vstack`,
  `cv2.findContours` results read as `[-2:]` / `[-2]`, `vor.points.ptp()` -> `np.ptp(vor.points)`.
* `writeGwlForRegex(..., writeHeader=True)`: new optional argument to leave out the TipSlicer header.
* `stackToGwl(..., fieldMode=False)` / `writeGwlFile(..., fieldMode=False)` and the new `writeGwlFieldFile`:
  optional DeScribe-style field file (only the writing, galvo coordinates relative to the field centre, laser
  parameters from the job file). The defaults leave the original behaviour unchanged.
* `stackToGwl(..., ZscalerFunc=None)` (passed on to `writeGwlFile`, as in TipSlicer's own "Updated" version) and, for the
  field files, a `MultLaserPower` line whenever that factor changes along z (the height-dependent laser power of the
  Adaptive Stitching notebook). Without a `ZscalerFunc` the field files are unchanged.
* `stlToStack(..., writingMask=...)` also accepts a function of z (block shear: the block outline moves with the height). An
  array behaves as before.
* `stlToStack(..., inFillMaster=...)`: when the erosion of the in-fill template makes a part vanish, the last erosion that still
  has every part is kept as the seed (the original kept the emptied one, so thin structures gave empty slices).
* the in-fill of `stlToStack` never erodes its seeds out of the block, and slices the solid when the fill finds nothing
  (a slice on a face); both gave empty slices for thin or plateau-like structures before.
