import Foundation
import Accelerate
import CoreML

struct BenchRow: Identifiable, Sendable {
    let id = UUID()
    let name: String
    let milliseconds: Double
    let rate: Double          // GFLOPS or GOPS
    let unit: String
}

struct BenchReport: Sendable {
    var rows: [BenchRow] = []
    var notes: [String] = []
}

/// Runs `body` repeatedly for at least `minSeconds` (after a warm-up) and returns seconds per call.
func measure(minSeconds: Double = 0.4, _ body: () -> Void) -> Double {
    body(); body()  // warm-up: page in buffers, spin the cores up
    var iterations = 0
    let start = sme_now_ns()
    var now = start
    repeat {
        body()
        iterations += 1
        now = sme_now_ns()
    } while Double(now - start) / 1e9 < minSeconds || iterations < 3
    return Double(now - start) / 1e9 / Double(iterations)
}

// MARK: - Raw GEMM benchmark

enum GEMMBenchmark {
    static func run(n: Int, useSME2: Bool) -> BenchReport {
        var report = BenchReport()
        let m32 = Int32(n), count = n * n
        let flops = 2.0 * Double(n) * Double(n) * Double(n)
        var rng = SplitMix64(seed: 7)

        // FP32 operands
        let a = (0..<count).map { _ in rng.nextFloat(in: -1...1) }
        let b = (0..<count).map { _ in rng.nextFloat(in: -1...1) }
        var ap = [Float](repeating: 0, count: sme_f32_packed_a_count(m32, m32))
        var bp = [Float](repeating: 0, count: sme_f32_packed_b_count(m32, m32))
        sme_f32_pack_a(a, m32, m32, &ap)
        sme_f32_pack_b(b, m32, m32, &bp)
        var cSME = [Float](repeating: 0, count: count)
        var cNEON = [Float](repeating: 0, count: count)
        var cACC = [Float](repeating: 0, count: count)

        // INT8 operands
        let a8 = (0..<count).map { _ in Int8.random(in: -127...127, using: &rng) }
        let b8 = (0..<count).map { _ in Int8.random(in: -127...127, using: &rng) }
        var ap8 = [Int8](repeating: 0, count: sme_s8_packed_a_bytes(m32, m32))
        var bp8 = [Int8](repeating: 0, count: sme_s8_packed_b_bytes(m32, m32))
        sme_s8_pack_a(a8, m32, m32, &ap8)
        sme_s8_pack_b(b8, m32, m32, &bp8)
        var c8SME = [Int32](repeating: 0, count: count)
        var c8NEON = [Int32](repeating: 0, count: count)

        func add(_ name: String, _ unit: String, _ seconds: Double) {
            report.rows.append(BenchRow(name: name, milliseconds: seconds * 1e3, rate: flops / seconds / 1e9, unit: unit))
        }

        if useSME2 {
            add("SME2 fp32 · FMOPA", "GFLOPS", measure { sme_f32_gemm_packed(m32, m32, m32, ap, bp, nil, 0, &cSME, m32) })
        }
        add("NEON fp32 · FMLA", "GFLOPS", measure { neon_f32_gemm_packed(m32, m32, m32, ap, bp, nil, 0, &cNEON, m32) })
        add("Accelerate fp32 · sgemm", "GFLOPS", measure {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m32, m32, m32, 1, a, m32, b, m32, 0, &cACC, m32)
        })
        if useSME2 {
            add("SME2 int8 · SMOPA", "GOPS", measure { sme_s8_gemm_packed(m32, m32, m32, ap8, bp8, &c8SME, m32) })
        }
        add("NEON int8 · SDOT", "GOPS", measure { neon_s8_gemm_packed(m32, m32, m32, ap8, bp8, &c8NEON, m32) })

        // Correctness: compare against Accelerate (fp32) and the NEON SDOT result (int8, exact).
        let neonErr = maxAbsDiff(cNEON, cACC)
        report.notes.append(String(format: "NEON vs Accelerate max |Δ| = %.2e", neonErr))
        if useSME2 {
            report.notes.append(String(format: "SME2 vs Accelerate max |Δ| = %.2e", maxAbsDiff(cSME, cACC)))
            let mismatches = zip(c8SME, c8NEON).filter { $0 != $1 }.count
            report.notes.append(mismatches == 0 ? "int8: SME2 matches NEON exactly" : "int8: \(mismatches) mismatches!")
        }
        return report
    }
}

func maxAbsDiff(_ x: [Float], _ y: [Float]) -> Float {
    zip(x, y).reduce(0) { max($0, abs($1.0 - $1.1)) }
}

// MARK: - Core ML / Neural Engine benchmark

/// Times the same-shaped MLP on the Neural Engine via Core ML (the only public path to the NPU).
/// Uses the bundled `MLPBench.mlpackage` (random weights, so it is a timing comparison only, not a
/// correctness check against the SME2/NEON/Accelerate outputs).
enum CoreMLBench {
    struct Result { var row: BenchRow?; var note: String }

    static func run(batch: Int, inputSize: Int, flopsPerBatch: Double) -> Result {
        guard let url = Bundle.main.url(forResource: "MLPBench", withExtension: "mlmodelc") else {
            return Result(row: nil, note: "Neural Engine: MLPBench model not in bundle — generate it with Tools/export_coreml_mlp.py.")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let model: MLModel
        let input: MLDictionaryFeatureProvider
        do {
            model = try MLModel(contentsOf: url, configuration: config)
            let arr = try MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: inputSize)], dataType: .float32)
            let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: batch * inputSize)
            var rng = SplitMix64(seed: 123)
            for i in 0..<(batch * inputSize) { ptr[i] = rng.nextFloat(in: -1...1) }
            input = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: arr)])
            _ = try model.prediction(from: input)   // validate the shape once; also warms the ANE up
        } catch {
            return Result(row: nil, note: "Neural Engine: Core ML prediction failed (\(error.localizedDescription)).")
        }
        let seconds = measure { _ = try? model.prediction(from: input) }
        let row = BenchRow(name: "Neural Engine (Core ML)", milliseconds: seconds * 1e3,
                           rate: flopsPerBatch / seconds / 1e9, unit: "GFLOPS")
        return Result(row: row, note: "Neural Engine row: Core ML computeUnits=.cpuAndNeuralEngine, separate random-weight model " +
                      "(timing only). Falls back to CPU where no ANE is available, e.g. the Simulator.")
    }
}

// MARK: - Neural-network inference benchmark

/// Owns the network off the main thread.
actor InferenceEngine {
    /// Stand-in for a pose-sequence classifier: 30 frames x 17 keypoints x 2 coords ≈ 1024 inputs.
    static let layerSizes = [1024, 2048, 2048, 1024, 16]

    private var network: DenseNetwork?

    func run(batch: Int, useSME2: Bool) -> BenchReport {
        let net: DenseNetwork
        if let existing = network {
            net = existing
        } else {
            net = DenseNetwork(sizes: Self.layerSizes)
            network = net
        }

        var report = BenchReport()
        var rng = SplitMix64(seed: 99)
        let input = (0..<(batch * net.inputSize)).map { _ in rng.nextFloat(in: -1...1) }
        let flops = net.flopsPerSample * Double(batch)

        var outputs: [Backend: [Float]] = [:]
        let backends = Backend.allCases.filter { useSME2 || $0 != .sme2 }
        input.withUnsafeBufferPointer { inBuf in
            for backend in backends {
                let seconds = measure { _ = net.forward(inBuf.baseAddress!, batch: batch, backend: backend) }
                let out = net.forward(inBuf.baseAddress!, batch: batch, backend: backend)
                outputs[backend] = Array(UnsafeBufferPointer(start: out, count: batch * net.outputSize))
                report.rows.append(BenchRow(name: backend.rawValue, milliseconds: seconds * 1e3,
                                            rate: flops / seconds / 1e9, unit: "GFLOPS"))
            }
        }

        // Neural Engine (NPU) via Core ML, on a separately bundled model of the same shape.
        let coreml = CoreMLBench.run(batch: batch, inputSize: net.inputSize, flopsPerBatch: flops)
        if let row = coreml.row { report.rows.append(row) }
        report.notes.append(coreml.note)

        report.notes.append("\(net.layers.count) dense layers, \(net.parameterCount.formatted()) parameters, " +
                            String(format: "%.2f GFLOP per batch", flops / 1e9))
        if let sme = outputs[.sme2], let ref = outputs[.accelerate] {
            let classes = net.outputSize
            var agree = 0
            for s in 0..<batch {
                let a = sme[(s * classes)..<((s + 1) * classes)]
                let r = ref[(s * classes)..<((s + 1) * classes)]
                if a.indices.max(by: { a[$0] < a[$1] }) == r.indices.max(by: { r[$0] < r[$1] }) { agree += 1 }
            }
            report.notes.append(String(format: "SME2 vs Accelerate: max |Δ| = %.2e, top-1 agreement %d/%d",
                                       maxAbsDiff(sme, ref), agree, batch))
        }
        if useSME2 && batch < DeviceInfo.current.fp32Lanes * 2 {
            report.notes.append("Batch \(batch) uses \(batch) of \(DeviceInfo.current.fp32Lanes * 2) rows in each ZA block, " +
                                "so most of the outer-product work is padding. SME pays off with batched or matrix-shaped work.")
        }
        return report
    }
}
