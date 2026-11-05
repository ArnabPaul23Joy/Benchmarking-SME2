#!/bin/bash
# Measure per-unit power on an Apple-silicon Mac (M4 or later) with powermetrics.
#
# SME2/NEON/Accelerate run on the CPU complex, so they show up under CPU-cluster power
# (P-/E-Cluster). The ANE path shows up under ANE Power (a separate rail). Combine the
# average power here with seconds/inference from the app's benchmark to get joules/inference.
#
# Usage:  sudo Tools/measure_power.sh sme2|neon|accelerate|ane [seconds]
#   (sudo is required: powermetrics needs root.)
#
# Prereq: build the sustained load generator to /tmp/sustained (see Tools/sustained_main.swift
# header for the exact swiftc command). For the ANE unit, also place MLPBench.mlmodelc next to it.

set -euo pipefail
UNIT="${1:-sme2}"
SECONDS_TO_RUN="${2:-20}"
BIN="/tmp/sustained"

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not found. Build it first — see Tools/sustained_main.swift header." >&2
  exit 1
fi

echo "Starting sustained $UNIT load…"
"$BIN" "$UNIT" >/dev/null 2>&1 &
LOAD_PID=$!
trap 'kill "$LOAD_PID" 2>/dev/null || true' EXIT
sleep 2  # let the unit ramp up before sampling

echo "Sampling power for ${SECONDS_TO_RUN}s (unit: $UNIT)…"
powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 1000 -n "$SECONDS_TO_RUN" \
  | grep -E 'Cluster Power|ANE Power|GPU Power|CPU die temperature|pressure' || true

echo "Done. CPU-cluster power ≈ SME2/NEON/Accelerate; ANE Power ≈ ANE path."
