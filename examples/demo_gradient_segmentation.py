"""Demo of the height-gradient-guided segmentation on a phase-height map.

    python examples/demo_gradient_segmentation.py [HEIGHT_MAP.png]

The default input is examples/demo_height_map.png: a 6-level phase-height map, 2 um pixels, 1 um peak height. Writes to
examples/demo_out/:
    gradient.png          the height-gradient map that guides the borders (|grad H|, 0 ... 1)
    labels.png            the block id of every pixel (16 bit: id + 1, 0 = outside the aperture)
    overlay.png           the block borders over the phase-height map
    block_<id>.png        the binary mask of a few blocks (what the slicer writes for that field)
    comparison.png        standard | footprint-edge | threshold | height-gradient-guided | phase-quality-guided, + metrics
    writing_path.png      the writing order of the phase-quality-guided mode (green = first, magenta = last)
    overlay_phase_quality.png   the phase-quality-guided block borders over the phase-height map
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gradient_segmentation as gs   # noqa: E402
import slicer_preview as sp          # noqa: E402
from recipes import default_recipe   # noqa: E402

PIXEL_UM, HEIGHT_MAX_UM, LEVELS, BASE_UM = 2.0, 1.0, 6, 0.2
src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("demo_height_map.png")
out = Path(__file__).with_name("demo_out")
out.mkdir(exist_ok=True)

grey = np.asarray(Image.open(src).convert("L"), dtype=float)
H = np.round(grey / grey.max() * (LEVELS - 1)) / (LEVELS - 1) * HEIGHT_MAX_UM            # quantized heights, um; row 0 = +y
rec = default_recipe()

# 1. the height-gradient-guided segmentation itself (on the grid with y growing with the row, as the slicer works)
Hy = np.flipud(H) + BASE_UM
centres, labels, grad = gs.segment(Hy, PIXEL_UM, rec.adaptive_radius)
print(f"{src.name}: {H.shape[1]} x {H.shape[0]} px, {len(centres)} blocks, write radius {rec.adaptive_radius:g} um")
Image.fromarray(np.round(np.flipud(grad) * 255).astype(np.uint8)).save(out / "gradient.png")
gs.save_label_image(labels, out / "labels.png")
for k in (0, len(centres) // 2, len(centres) - 1):                                          # per-block binary masks
    Image.fromarray((np.flipud(gs.block_mask(labels, k)) * 255).astype(np.uint8)).save(out / f"block_{k}.png")

# 2. the preview with the borders over the phase-height map, and the comparison of the five modes
fp = sp.design_footprint(H, PIXEL_UM, BASE_UM, True, max_side=1100, segment_height_um=BASE_UM)
lay = sp.compute_layout(fp, "adaptive", rec, segmentation_mode="height_gradient_guided")
sp.render(lay, rec.overlap).save(out / "overlay.png")
img, lays = sp.comparison_image(H, PIXEL_UM, BASE_UM, True, rec, BASE_UM, modes=("standard", "footprint", "threshold", "gradient", "phase"))
img.save(out / "comparison.png")
sp.render_path(lays["phase"]).save(out / "writing_path.png")
sp.render(lays["phase"], rec.overlap, path=False).save(out / "overlay_phase_quality.png")

# 3. the metrics of the five (same grid, same seam cost, so they compare)
met = sp.layout_metrics(lays)
print(f"{'mode':10s} {'blocks':>6s} {'seam mm':>8s} {'mean cost':>9s} {'on a step':>9s} {'islands':>8s} {'travel mm':>10s}")
for key, m in met.items():
    print(f"{key:10s} {m['blocks']:6d} {m['seam_length_um'] / 1000:8.1f} {m['seam_cost_mean']:9.3f} {m['seam_on_step'] * 100:8.1f}% "
          f"{m['extra_pieces']:8d} {m['travel_um'] / 1000:10.1f}")
print("wrote", ", ".join(sorted(p.name for p in out.iterdir())), "to", out)
