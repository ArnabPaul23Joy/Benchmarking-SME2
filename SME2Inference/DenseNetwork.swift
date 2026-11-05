import Foundation
import Accelerate

/// Which engine executes the matrix multiplies.
enum Backend: String, CaseIterable, Identifiable, Sendable {
    case sme2 = "SME2 (ZA outer products)"
    case neon = "NEON (non-streaming)"
    case accelerate = "Accelerate (cblas_sgemm)"
    var id: String { rawValue }
}

/// One fully connected layer: y = x·W + b, optional ReLU.
/// Weights are stored K x N (in x out), row-major, and pre-packed for the SME/NEON kernels.
final class DenseLayer {
    let inFeatures: Int
    let outFeatures: Int
    let relu: Bool
    let weights: UnsafeMutablePointer<Float>        // inFeatures * outFeatures
    let bias: UnsafeMutablePointer<Float>           // outFeatures
    let packedWeights: UnsafeMutablePointer<Float>  // panel layout used by the kernels

    init(inFeatures: Int, outFeatures: Int, relu: Bool, weights w: [Float], bias b: [Float]) {
        precondition(w.count == inFeatures * outFeatures && b.count == outFeatures)
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.relu = relu
        weights = .allocate(capacity: w.count)
        weights.initialize(from: w, count: w.count)
        bias = .allocate(capacity: b.count)
        bias.initialize(from: b, count: b.count)
        let packedCount = sme_f32_packed_b_count(Int32(inFeatures), Int32(outFeatures))
        packedWeights = .allocate(capacity: packedCount)
        // Weights are packed once at load time; only activations are packed per inference.
        sme_f32_pack_b(weights, Int32(inFeatures), Int32(outFeatures), packedWeights)
    }

    deinit {
        weights.deallocate()
        bias.deallocate()
        packedWeights.deallocate()
    }
}

/// A small MLP inference engine whose matmuls run on SME2, NEON or Accelerate.
final class DenseNetwork: @unchecked Sendable {
    let layers: [DenseLayer]

    private var capacityBatch = 0
    private var ping: UnsafeMutablePointer<Float>?
    private var pong: UnsafeMutablePointer<Float>?
    private var packedActivations: UnsafeMutablePointer<Float>?

    init(layers: [DenseLayer]) {
        precondition(!layers.isEmpty)
        for (a, b) in zip(layers, layers.dropFirst()) {
            precondition(a.outFeatures == b.inFeatures, "layer shapes do not chain")
        }
        self.layers = layers
    }

    /// Random (He-uniform) weights. Good enough for benchmarking the inference path.
    convenience init(sizes: [Int], seed: UInt64 = 42) {
        var rng = SplitMix64(seed: seed)
        var layers: [DenseLayer] = []
        for i in 0..<(sizes.count - 1) {
            let k = sizes[i], n = sizes[i + 1]
            let limit = Float((6.0 / Double(k)).squareRoot())
            let w = (0..<(k * n)).map { _ in rng.nextFloat(in: -limit...limit) }
            let b = (0..<n).map { _ in rng.nextFloat(in: -0.05...0.05) }
            layers.append(DenseLayer(inFeatures: k, outFeatures: n, relu: i < sizes.count - 2, weights: w, bias: b))
        }
        self.init(layers: layers)
    }

    /// Loads weights exported by `Tools/export_mlp.py` (format described there).
    convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        var offset = 0
        func readU32() throws -> UInt32 {
            guard offset + 4 <= data.count else { throw CocoaError(.fileReadCorruptFile) }
            let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            return UInt32(littleEndian: v)
        }
        func readFloats(_ count: Int) throws -> [Float] {
            guard offset + count * 4 <= data.count else { throw CocoaError(.fileReadCorruptFile) }
            let out = data.withUnsafeBytes { raw in
                (0..<count).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + $0 * 4, as: UInt32.self))) }
            }
            offset += count * 4
            return out
        }
        guard try readU32() == 0x5745_4D53 else { throw CocoaError(.fileReadCorruptFile) }  // "SMEW"
        let count = Int(try readU32())
        var layers: [DenseLayer] = []
        for _ in 0..<count {
            let k = Int(try readU32()), n = Int(try readU32()), relu = try readU32() != 0
            let w = try readFloats(k * n)
            let b = try readFloats(n)
            layers.append(DenseLayer(inFeatures: k, outFeatures: n, relu: relu, weights: w, bias: b))
        }
        self.init(layers: layers)
    }

    deinit { freeBuffers() }

    var inputSize: Int { layers[0].inFeatures }
    var outputSize: Int { layers[layers.count - 1].outFeatures }
    var flopsPerSample: Double { layers.reduce(0) { $0 + 2.0 * Double($1.inFeatures * $1.outFeatures) } }
    var parameterCount: Int { layers.reduce(0) { $0 + $1.inFeatures * $1.outFeatures + $1.outFeatures } }

    /// Runs `batch` samples (row-major, batch x inputSize). The returned pointer is valid
    /// until the next call to `forward`.
    func forward(_ input: UnsafePointer<Float>, batch: Int, backend: Backend) -> UnsafePointer<Float> {
        ensureCapacity(batch)
        var src = input
        let m = Int32(batch)
        for (index, layer) in layers.enumerated() {
            let dst = (index % 2 == 0 ? ping : pong)!
            let k = Int32(layer.inFeatures), n = Int32(layer.outFeatures)
            switch backend {
            case .sme2:
                sme_f32_pack_a(src, m, k, packedActivations!)
                sme_f32_gemm_packed(m, n, k, packedActivations!, layer.packedWeights, layer.bias, layer.relu ? 1 : 0, dst, n)
            case .neon:
                sme_f32_pack_a(src, m, k, packedActivations!)
                neon_f32_gemm_packed(m, n, k, packedActivations!, layer.packedWeights, layer.bias, layer.relu ? 1 : 0, dst, n)
            case .accelerate:
                for row in 0..<batch {
                    (dst + row * layer.outFeatures).update(from: layer.bias, count: layer.outFeatures)
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                            1, src, k, layer.weights, n, 1, dst, n)
                if layer.relu {
                    var zero: Float = 0
                    vDSP_vthres(dst, 1, &zero, dst, 1, vDSP_Length(batch * layer.outFeatures))
                }
            }
            src = UnsafePointer(dst)
        }
        return src
    }

    private func ensureCapacity(_ batch: Int) {
        guard batch > capacityBatch else { return }
        freeBuffers()
        let widest = layers.map { max($0.inFeatures, $0.outFeatures) }.max()!
        ping = .allocate(capacity: widest * batch)
        pong = .allocate(capacity: widest * batch)
        let packed = layers.map { sme_f32_packed_a_count(Int32(batch), Int32($0.inFeatures)) }.max()!
        packedActivations = .allocate(capacity: packed)
        capacityBatch = batch
    }

    private func freeBuffers() {
        ping?.deallocate(); pong?.deallocate(); packedActivations?.deallocate()
        ping = nil; pong = nil; packedActivations = nil
        capacityBatch = 0
    }
}

/// Small deterministic RNG so every run uses the same weights.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        Float.random(in: range, using: &self)
    }
}
