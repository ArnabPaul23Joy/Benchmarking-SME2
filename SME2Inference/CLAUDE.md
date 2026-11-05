# SME2Inference

iOS app (SwiftUI + C) that runs matrix multiplies and small neural networks directly on the Arm SME2 unit of Apple CPUs. The target device is the iPhone 17 Pro (A19 Pro).

**Why it exists:** the owner profiled `yolo26n-pose` and `ExerciseClassifier` under Core ML on CPU, GPU and Neural Engine, and asked whether models can run on SME2. Core ML has no SME compute unit, so this app targets SME2 with hand-written kernels.

## Current status (2026-11-03)

Tested on an M4 Mac, which has SME2 and a 512-bit streaming vector length:
- The kernels match the scalar reference at all test sizes, including odd ones.
- The GEMM and inference benchmark code passes when run as a macOS CLI build. Tested at n=1024 and batch 1, 32 and 128.
- The app builds for iOS with no warnings, and the binary contains `smstart`, `fmopa` and `smopa`.
- The app launches in the iPhone 17 Pro Simulator and the first screen renders.

Not tested yet:
- **Running on a real iPhone 17 Pro. This is the next step.** SME2 and the vector length on A19 Pro are expected but unconfirmed; the app's "This device" section reports both.
- Tapping the Run buttons in the app. Only the code behind them was run, through the CLI build.
- `DenseNetwork(contentsOf:)` and `Tools/export_mlp.py`. Neither has ever been run.
- The NEON and Accelerate paths on a CPU without SME.
- The 2048² matrix size and batch size 8.

## Layout

| Path | Purpose |
|---|---|
| `SME2Inference/Kernels/sme_kernels.c`, `.h` | SME2 fp32 (`FMOPA`) and int8 (`SMOPA`) GEMMs, NEON baselines, packing, scalar references, CPU feature detection |
| `SME2Inference/DenseNetwork.swift` | MLP engine with SME2, NEON and Accelerate backends, plus the weight loader |
| `SME2Inference/Benchmarks.swift` | GEMM benchmark, `InferenceEngine` actor, and `CoreMLBench` (Neural Engine timing) |
| `SME2Inference/ContentView.swift` | UI and `AppModel` |
| `SME2Inference/ThermalStress.swift` | Per-unit sustained-load / thermal throttle benchmark (SME2, NEON, ANE) and `ProcessInfo.thermalState` helpers |
| `SME2Inference/MLPBench.mlpackage` | Core ML version of the MLP, bundled so inference can be timed on the Neural Engine. Generated, random weights (timing only). Regenerate with `Tools/export_coreml_mlp.py` |
| `Tests/kernel_test.c` | Mac test and benchmark for the kernels (outside the app target) |
| `Tools/export_mlp.py` | Exports a PyTorch MLP to the `.smew` format |
| `Tools/export_coreml_mlp.py` | Exports the same MLP to `MLPBench.mlpackage` (needs torch + coremltools) |
| `Tools/sustained_main.swift` | Mac-only single-unit flat-out load generator for power measurement (outside the app target) |
| `Tools/measure_power.sh` | Mac-only `powermetrics` wrapper that reports per-unit power (CPU cluster vs ANE rail) |
| `data.txt` | Collected benchmark runs (GEMM, inference, thermal stress) with plain-language interpretations. Device unconfirmed (likely M4) |

## Commands

Run from this directory. Built with Xcode 26.6 and the iOS 26.5 SDK.

```sh
# Kernel correctness and benchmark. Needs an SME2 Mac (M4 or later): on other CPUs the SME
# kernels return without writing, so the checks fail.
xcrun clang -O3 -framework Accelerate -I SME2Inference/Kernels \
  SME2Inference/Kernels/sme_kernels.c Tests/kernel_test.c -o /tmp/kt && /tmp/kt

# Device build without signing
xcodebuild -scheme SME2Inference -configuration Release -sdk iphoneos \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build

# Confirm SME instructions in the built binary (BIN = path to the app executable)
xcrun objdump -d --mattr=+sme2 "$BIN" | grep -E 'smstart|fmopa|smopa'
```

To run the Swift benchmark code on a Mac, write a `main.swift` that calls `GEMMBenchmark.run(n:useSME2:)` and `InferenceEngine().run(batch:useSME2:)` and prints the reports. That harness isn't checked in. Then:

```sh
xcrun clang -O3 -c SME2Inference/Kernels/sme_kernels.c -o /tmp/k.o
xcrun swiftc -O -Xcc -DACCELERATE_NEW_LAPACK \
  -import-objc-header SME2Inference/SME2Inference-Bridging-Header.h \
  SME2Inference/DeviceInfo.swift SME2Inference/DenseNetwork.swift SME2Inference/Benchmarks.swift \
  main.swift /tmp/k.o -o /tmp/hostbench
```

## Things that will bite you

- **`CLANG_ENABLE_MODULES` must stay `NO`.** With modules on, `arm_sme.h` fails with "Module '_Builtin_intrinsics.arm.sve' requires feature 'sve'".
- **Use per-function target attributes, not global flags.** Kernels use `__attribute__((target("sme2")))` with `__arm_new("za") __arm_locally_streaming`, so the rest of the app compiles normally.
- **Streaming code must stay self-contained.** Don't call normal functions (`memcpy`, `printf`) or use NEON inside a streaming function. The M4 doesn't report `FEAT_SME_FA64`, so most NEON instructions are illegal in streaming mode. Do packing outside the kernel, and use macros rather than helper functions, because tile numbers must be compile-time constants.
- **`RDSVL` crashes (SIGILL) on CPUs without SME.** Check `sme_cpu_has_sme()` before calling `svcntsw()`. Calling `svcntsw()` also needs `target("sme")`, even outside streaming mode.
- **Files are auto-included.** The project was written by hand and uses a file-system-synchronized group, so new files under `SME2Inference/` join the target without editing `project.pbxproj`.
- **Signing:** `DEVELOPMENT_TEAM = 794SW6863L`, bundle ID `arnabksu.SME2Inference`. Change these if the Mac uses a different Apple account.
- **Use Release for timings.** C is `-O3` in both configurations, but Swift is `-Onone` in Debug.
- **`hw.machine` returns `arm64` on a Mac and in the Simulator.** Only a real device shows `iPhoneXX,X`.

## Baseline results (M4, single core, Release)

| Test | SME2 | NEON | Accelerate |
|---|---|---|---|
| fp32 GEMM, 1024² | 1852 GFLOPS | 110 GFLOPS | 1863 GFLOPS |
| int8 GEMM, 1024² | 3610 GOPS | 470 GOPS | n/a |
| MLP, batch 1 | 0.666 ms | 1.393 ms | 0.567 ms |
| MLP, batch 32 | 0.908 ms | 5.845 ms | 1.282 ms |
| MLP, batch 128 | 2.421 ms | 22.668 ms | 2.512 ms |

The MLP is 1024→2048→2048→1024→16 with random weights.

For comparison, the owner's Core ML CPU-only trace on iPhone averaged 17.44 ms per prediction for `yolo26n-pose` and 130.78 µs for `ExerciseClassifier`. The trace and CSV files are in the parent folder, if it was copied along; the model files are not.

## What the results mean

- Accelerate's fp32 GEMM output was identical to the SME2 kernel's at 1024², at the same speed. That strongly suggests Accelerate uses SME internally, though Apple doesn't document it. Core ML's CPU path may therefore already use SME too, which is unconfirmed.
- SME pays off on batched, matrix-shaped work. At batch 1, only 1 of the 32 rows in each ZA block is real work, and Accelerate wins.
- Don't expect multithreading to scale, because each CPU cluster shares one SME unit.

## How we know SME2 actually runs (not just "it's fast")

Three independent checks, strongest first. These prove *our* kernel runs on the SME unit; Accelerate using SME remains a strong inference (identical output + speed), not confirmed.

1. **The instructions are in the binary.** `xcrun objdump -d --mattr=+sme2 "$BIN" | grep -E 'smstart|fmopa|smopa'` finds the actual matrix opcodes the compiler emitted. They can't execute unless the hardware runs them.
2. **The code is gated and still produced correct output.** `sme_*_gemm_packed()` returns early (leaving `C` all zeros) when `sme_cpu_has_sme2()` is false (`sme_kernels.c:203`, `:359`). The benchmark's `SME2 vs Accelerate max |Δ| = 0.00` means the SME2 buffer exactly equals Accelerate's correct answer — impossible if the kernel hadn't run. The app's "This device" screen shows `FEAT_SME2 = Yes` for the same reason.
3. **The speed is physically impossible on NEON.** SME2 and NEON run the same packed data through the same loop; only `FMOPA` vs `FMLA` differs. NEON on one core tops out ~110 GFLOPS, so ~1700 GFLOPS can only come from the matrix unit.
4. *(optional, live)* Instruments → CPU Counters with SME PMU events spikes only during the SME2 rows. Running it during the Accelerate row would upgrade "Accelerate uses SME" from inferred to confirmed.

## Measuring thermal & energy signatures

Not implemented in the app yet — this is the agreed methodology (2026-11-04). Energy and thermal are different questions needing different tools.

- **Framing.** Energy = avg power × time, so the number that matters is **joules per inference** (report as inferences/joule), not watts — a higher-power kernel that finishes faster can still win ("race to idle"). Thermal is sustained power over time and only appears under a long run. Caveat: the SME unit sits **inside the CPU complex**, so no tool isolates "SME-only" power; compare the CPU cluster running SME2 vs the same cluster running NEON, and vs the ANE running Core ML. Use the *deltas*.
- **Energy, quantitative (M4 Mac).** `sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 1000 -n 60` while a CLI harness loops ONE backend for ~60 s. SME2/NEON/Accelerate appear under CPU-cluster power; the Core ML path appears under **ANE power** (separate rail) — so this directly compares SME2 vs NPU efficiency. Energy/inference = avg_power_W × seconds/inference (seconds/inference comes from `measure`). Expect SME2 ≪ NEON in joules despite higher peak power; int8/SMOPA most efficient per op; ANE may beat SME2 on joules even where SME2 wins on ms.
- **Energy, device (iPhone).** No `powermetrics`. Options, roughly decreasing fidelity: Instruments Energy/Power template, Xcode debug-navigator Energy gauge (coarse), battery-%-drain over a few minutes per backend, MetricKit `MXCPUMetric` cumulativeCPUTime/instructions as a proxy (field data, not joules).
- **Thermal, device.** `ProcessInfo.processInfo.thermalState` (nominal→fair→serious→critical) + `thermalStateDidChangeNotification` — official, on-device. Throttle curve = loop a backend for minutes, log GFLOPS + thermalState every second; GFLOPS falling as thermalState rises is the signature. Mac `powermetrics` also reports die temp / thermal pressure. Xcode "Test under adverse device conditions" can force thermal states.
- **App support (built 2026-11-04).** Hard limit first: **no public iOS API exposes per-unit power (watts/joules) at runtime** — `powermetrics` is macOS/root only; MetricKit gives aggregated field CPU time, not live per-unit watts. So the split is: *thermal on-device, power on Mac*. Implemented:
  - `ThermalStress.swift` + a "Thermal signature (per unit)" UI section: drives SME2, NEON and ANE at full load for a chosen duration (10/30/60/120 s), sampling throughput + `thermalState` each second → the per-unit throttle curve (peak→end GFLOPS, % throttle, thermal start→max). Logged to console and shown on screen. SME2/NEON share a 1024² fp32 GEMM; ANE runs the Core ML MLP at batch 32.
  - `thermalState` added to the existing GEMM/inference console log headers.
  - Mac power path: `Tools/sustained_main.swift` (flat-out single-unit load generator) + `Tools/measure_power.sh` (`sudo` powermetrics wrapper). CPU-cluster power ≈ SME2/NEON/Accelerate; ANE Power rail ≈ ANE. Combine avg power with the app's seconds/inference to get joules/inference. Not yet run (needs the host build + sudo).

## Next steps

1. Run on an iPhone 17 Pro and record the results here next to the M4 numbers.
2. To run `yolo26n-pose` on SME2, use ExecuTorch with the XNNPACK backend and KleidiAI SME2 kernels, and compare against the Core ML traces. This was proposed but not started. Don't hand-write every YOLO layer.
3. Test the `.smew` export and load path with a real model.

## Session log

Compact record of assistant-assisted changes and findings. Newest first.

### 2026-11-04

- **Per-unit thermal + power measurement implemented.** On-device thermal: new `ThermalStress.swift` and "Thermal signature (per unit)" section stress SME2 / NEON / ANE in isolation and record the throughput throttle curve + `thermalState` over 10–120 s (console + UI). Mac power: `Tools/sustained_main.swift` + `Tools/measure_power.sh` wrap `powermetrics` (CPU-cluster power for SME2/NEON, ANE rail for the Neural Engine). **Hard limit recorded:** no public iOS API gives per-unit watts, so real power numbers only come from the Mac path; the phone gives thermal + sustained throughput. Builds clean; thermal test not yet run on device, Mac power path not yet run (needs sudo + host build).
- **Thermal/energy methodology agreed and recorded** earlier the same day (see "Measuring thermal & energy signatures"): measure joules/inference (race-to-idle), not watts.

### 2026-11-03

- **NPU (Neural Engine) measurement added.** The inference benchmark now includes a `Neural Engine (Core ML)` row. Apple exposes the ANE only through Core ML, which needs a bundled model — so `Tools/export_coreml_mlp.py` generates `MLPBench.mlpackage` (same 1024→2048→2048→1024→16 shape, flexible batch 1–128, random weights), the synchronized group makes Xcode compile+bundle it as `MLPBench.mlmodelc`, and `CoreMLBench.run` (in `Benchmarks.swift`) times `model.prediction` with `computeUnits = .cpuAndNeuralEngine`. Timing-only (weights differ from the SME2/NEON/Accelerate net, so no correctness compare). Falls back to CPU where no ANE exists (e.g. Simulator). Verified: builds clean, `MLPBench.mlmodelc` is in the device app bundle. Not yet run on device.
- **Console logging added.** `BenchReport.log(title:)` extension in `ContentView.swift` prints each row (name, ms, rate, unit) and all notes to the Xcode console; `runGEMM()`/`runInference()` call it with a title like `GEMM benchmark · 1024² · SME2=true`. Lets us read results without reading the UI.
- **First per-size GEMM run captured** (device unconfirmed — M4 or iPhone 17 Pro; `SME2=true`). fp32 SME2 GFLOPS by size: 256²=1251, 512²=1560, 1024²=1720, 2048²=1506. int8 SME2 GOPS: 256²=2379, 512²=3326, 1024²=3844, 2048²=3915. SME2≈Accelerate at every size; NEON ~16–19× slower; all correctness deltas 0.00. *Confirm the device and, if it's the iPhone 17 Pro, promote these into the baseline table.*
- **First inference run captured** (same device, `SME2=true`, MLP 1024→2048→2048→1024→16, 8.41M params). SME2 ms / GFLOPS by batch: 1 → 0.639 / 26, 8 → 0.568 / 237, 32 → 0.705 / 763, 128 → 2.041 / 1054. Accelerate ms: 0.458, 0.894, 1.126, 2.347. NEON flat ~100 GFLOPS throughout. top-1 agreement N/N every batch; max |Δ| ≈ 3e-06.
- **Key inference insight — crossover at batch ~8.** The ZA block is 32 rows tall, so batch = rows used out of 32. Batch 1 wastes 31/32 on padding (~3% full) → Accelerate wins. Batch ≥8 → SME2 wins and the lead grows (batch 32: 1.6× faster than Accelerate; SME2 even beats Accelerate on these tall-skinny shapes). SME2 throughput scales ~40× from batch 1→128 purely from filling the block. **Rule: single-sample → use Accelerate/NEON; batched (8+) → SME2.** Sub-2ms times mean batch-1 GFLOPS understate the engine (fixed streaming-mode overhead dominates); batch 128 = honest sustained throughput.
- **Interpretation notes.** Performance peaks at 1024² then dips at 2048² — the matrices stop fitting in cache, so it becomes memory-bound (Accelerate's better cache-blocking is why it still leads at 2048²). int8 ≈ 2× fp32 because 8-bit values pack ~4× denser through the engine; exact NEON match confirms the int8 kernel (integer math has no rounding).
- **Added "How we know SME2 actually runs" section** (binary opcodes / zero-on-no-SME gating / NEON-impossible speed / Instruments). See above.
