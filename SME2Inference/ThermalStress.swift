import Foundation
import CoreML

// MARK: - Thermal state helpers

/// Thin wrapper over `ProcessInfo.thermalState` (the only on-device thermal signal Apple exposes).
/// Note: thermal state is system-wide, not per-unit; we attribute it to a unit by stressing that
/// unit in isolation and watching how its own throughput throttles.
enum ThermalLevel {
    static func current() -> Int { level(ProcessInfo.processInfo.thermalState) }

    static func level(_ s: ProcessInfo.ThermalState) -> Int {
        switch s {
        case .nominal: return 0
        case .fair: return 1
        case .serious: return 2
        case .critical: return 3
        @unknown default: return -1
        }
    }

    static func name(_ level: Int) -> String {
        switch level {
        case 0: return "nominal"
        case 1: return "fair"
        case 2: return "serious"
        case 3: return "critical"
        default: return "unknown"
        }
    }
}

// MARK: - Sustained-load (thermal/throttle) benchmark

/// One throughput + thermal reading taken during a sustained run.
struct StressSample: Sendable {
    let elapsed: Double   // seconds since this unit's run started
    let gflops: Double    // sustained throughput over the sample window
    let thermal: Int      // ThermalLevel at the end of the window
}

/// The throttle curve for one compute unit under sustained load.
struct StressSummary: Sendable, Identifiable {
    let id = UUID()
    let unit: String
    let available: Bool
    let samples: [StressSample]
    let note: String

    var startGFLOPS: Double { samples.first?.gflops ?? 0 }
    var endGFLOPS: Double { samples.last?.gflops ?? 0 }
    var peakGFLOPS: Double { samples.map(\.gflops).max() ?? 0 }
    /// How far throughput fell from its peak by the end — the thermal throttle signature.
    var throttlePercent: Double { peakGFLOPS > 0 ? (1 - endGFLOPS / peakGFLOPS) * 100 : 0 }
    var startThermal: Int { samples.first?.thermal ?? 0 }
    var maxThermal: Int { samples.map(\.thermal).max() ?? 0 }

    func log() {
        print("── Thermal stress · \(unit) ──")
        guard available else { print("  (skipped: \(note))"); return }
        for s in samples {
            print(String(format: "  t=%5.1fs  %7.0f GFLOPS  thermal=%@", s.elapsed, s.gflops, ThermalLevel.name(s.thermal)))
        }
        print(String(format: "  summary: peak %.0f → end %.0f GFLOPS (%.1f%% throttle), thermal %@ → %@",
                     peakGFLOPS, endGFLOPS, throttlePercent, ThermalLevel.name(startThermal), ThermalLevel.name(maxThermal)))
    }
}

enum ThermalStress {
    /// Drives `body` continuously for `seconds`, recording throughput + thermal state every `window` seconds.
    static func stress(unit: String, seconds: Double, window: Double = 1.0,
                       flopsPerIteration: Double, body: () -> Void) -> StressSummary {
        body()  // warm-up, not measured
        var samples: [StressSample] = []
        let start = sme_now_ns()
        var windowStart = start
        var windowIterations = 0
        while true {
            body()
            windowIterations += 1
            let now = sme_now_ns()
            let windowElapsed = Double(now - windowStart) / 1e9
            if windowElapsed >= window {
                let gflops = flopsPerIteration * Double(windowIterations) / windowElapsed / 1e9
                let elapsed = Double(now - start) / 1e9
                samples.append(StressSample(elapsed: elapsed, gflops: gflops, thermal: ThermalLevel.current()))
                windowStart = now
                windowIterations = 0
                if elapsed >= seconds { break }
            }
        }
        return StressSummary(unit: unit, available: true, samples: samples, note: "")
    }
}

// MARK: - Per-unit suite (SME2, NEON, ANE)

enum ThermalStressSuite {
    /// Runs a sustained load on each compute unit in turn and returns their throttle curves.
    /// Per-unit power (watts) is intentionally absent: no public iOS API exposes it — use
    /// `Tools/measure_power.sh` on an M4 Mac for real power numbers.
    static func run(seconds: Double, useSME2: Bool) -> [StressSummary] {
        var reports: [StressSummary] = []

        // Shared fp32 GEMM operands for the SME2 and NEON runs (same work, different unit).
        let n = 1024, m32 = Int32(1024)
        let gemmFlops = 2.0 * Double(n) * Double(n) * Double(n)
        var rng = SplitMix64(seed: 7)
        let a = (0..<(n * n)).map { _ in rng.nextFloat(in: -1...1) }
        let b = (0..<(n * n)).map { _ in rng.nextFloat(in: -1...1) }
        var ap = [Float](repeating: 0, count: sme_f32_packed_a_count(m32, m32))
        var bp = [Float](repeating: 0, count: sme_f32_packed_b_count(m32, m32))
        sme_f32_pack_a(a, m32, m32, &ap)
        sme_f32_pack_b(b, m32, m32, &bp)
        var c = [Float](repeating: 0, count: n * n)

        if useSME2 {
            reports.append(ThermalStress.stress(unit: "SME2 · fp32 GEMM 1024²", seconds: seconds, flopsPerIteration: gemmFlops) {
                sme_f32_gemm_packed(m32, m32, m32, ap, bp, nil, 0, &c, m32)
            })
        } else {
            reports.append(StressSummary(unit: "SME2", available: false, samples: [], note: "no SME2 on this device"))
        }

        reports.append(ThermalStress.stress(unit: "NEON · fp32 GEMM 1024²", seconds: seconds, flopsPerIteration: gemmFlops) {
            neon_f32_gemm_packed(m32, m32, m32, ap, bp, nil, 0, &c, m32)
        })

        reports.append(aneStress(seconds: seconds))
        return reports
    }

    /// Sustained Core ML inference on the Neural Engine.
    private static func aneStress(seconds: Double) -> StressSummary {
        let unit = "ANE · Core ML MLP (batch 32)"
        let batch = 32
        let sizes = InferenceEngine.layerSizes
        let flopsPerSample = zip(sizes, sizes.dropFirst()).reduce(0.0) { $0 + 2.0 * Double($1.0 * $1.1) }
        let flopsPerIteration = flopsPerSample * Double(batch)

        guard let url = Bundle.main.url(forResource: "MLPBench", withExtension: "mlmodelc") else {
            return StressSummary(unit: unit, available: false, samples: [], note: "MLPBench model not bundled")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model: MLModel
        let input: MLDictionaryFeatureProvider
        do {
            model = try MLModel(contentsOf: url, configuration: config)
            let arr = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: sizes[0])], dataType: .float32)
            let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: batch * sizes[0])
            var rng = SplitMix64(seed: 123)
            for i in 0..<(batch * sizes[0]) { ptr[i] = rng.nextFloat(in: -1...1) }
            input = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: arr)])
            _ = try model.prediction(from: input)
        } catch {
            return StressSummary(unit: unit, available: false, samples: [], note: "Core ML failed: \(error.localizedDescription)")
        }
        return ThermalStress.stress(unit: unit, seconds: seconds, flopsPerIteration: flopsPerIteration) {
            _ = try? model.prediction(from: input)
        }
    }
}
