#!/bin/bash
# Builds "Phase Mask Quantizer.app" (double-click launcher, no Terminal window) next to phase_mask_quantizer.py.
set -e
cd "$(dirname "$0")"
osacompile -o "../Phase Mask Quantizer.app" "Phase Mask Quantizer.applescript"
echo "Built ../Phase Mask Quantizer.app  (first launch: right-click > Open if macOS asks)"
