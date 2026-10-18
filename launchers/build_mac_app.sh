#!/bin/bash
# Builds "NanoSlicer.app" (double-click launcher, no Terminal window) next to nanoslicer.py.
set -e
cd "$(dirname "$0")"
osacompile -o "../NanoSlicer.app" "NanoSlicer.applescript"
echo "Built ../NanoSlicer.app  (first launch: right-click > Open if macOS asks)"
