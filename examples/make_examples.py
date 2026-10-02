"""Regenerate the example phase images in this folder:  python examples/make_examples.py

Both are 8-bit grayscale PNGs, black = 0, white = 2*pi, so they can be loaded directly.
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).parent
N = 256


def save(name, phase):
    wrapped = np.mod(phase, 2 * np.pi) / (2 * np.pi)
    Image.fromarray(np.round(wrapped * 255).astype(np.uint8)).save(OUT / name)


y, x = np.mgrid[-N // 2:N // 2, -N // 2:N // 2].astype(float)
r2 = x ** 2 + y ** 2

# Fresnel-style lens phase: -pi * r^2 / (lambda * f), here in pixel units
save("lens_phase.png", -np.pi * r2 / 900.0)

# Spiral phase plate (optical vortex, topological charge 3)
save("vortex_phase.png", 3 * np.arctan2(y, x))

# Smooth random-ish blob field (sum of a few sinusoids), good for checking quantization banding
save("smooth_phase.png", 2.5 * np.sin(x / 23) + 2.0 * np.cos(y / 17) + 1.5 * np.sin((x + y) / 31))
