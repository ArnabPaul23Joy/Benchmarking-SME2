// Sustained single-unit load generator for power measurement on a Mac.
//
// Runs ONE compute unit flat-out forever (until killed) so `powermetrics` can sample its power
// in a separate process. Pair it with Tools/measure_power.sh.
//
// Build (needs an SME2 Mac; see CLAUDE.md for the full host-build recipe):
//   xcrun clang -O3 -c SME2Inference/Kernels/sme_kernels.c -o /tmp/k.o
//   xcrun swiftc -O -Xcc -DACCELERATE_NEW_LAPACK \
//     -import-objc-header SME2Inference/SME2Inference-Bridging-Header.h \
//     SME2Inference/DeviceInfo.swift SME2Inference/DenseNetwork.swift \
//     SME2Inference/Benchmarks.swift Tools/sustained_main.swift /tmp/k.o -o /tmp/sustained
//
// Run:  /tmp/sustained sme2 | neon | accelerate | ane
//
// ane requires a compiled MLPBench.mlmodelc next to the binary; generate the .mlpackage with
// Tools/export_coreml_mlp.py and compile it with `xcrun coremlcompiler compile MLPBench.mlpackage .`.

import Foundation
import Accelerate
import CoreML

let unit = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "sme2"
let n = 1024, m32 = Int32(1024)
var rng = SplitMix64(seed: 7)

switch unit {
case "sme2", "neon":
    let a = (0..<(n * n)).map { _ in rng.nextFloat(in: -1...1) }
    let b = (0..<(n * n)).map { _ in rng.nextFloat(in: -1...1) }
    var ap = [Float](repeating: 0, count: sme_f32_packed_a_count(m32, m32))
    var bp = [Float](repeating: 0, count: sme_f32_packed_b_count(m32, m32))
    sme_f32_pack_a(a, m32, m32, &ap)
    sme_f32_pack_b(b, m32, m32, &bp)
    var c = [Float](repeating: 0, count: n * n)
    FileHandle.standardError.write(Data("sustained: \(unit) fp32 GEMM 1024², Ctrl-C to stop\n".utf8))
    while true {
        if unit == "sme2" {
            sme_f32_gemm_packed(m32, m32, m32, ap, bp, nil, 0, &c, m32)
        } else {
            neon_f32_gemm_packed(m32, m32, m32, ap, bp, nil, 0, &c, m32)
        }
    }

case "accelerate":
    let a = (0..<(n * n)).map { _ in rng.nextFloat(in: -1...1) }
    let b = (0..<(n * n)).map { _ in rng.nextFloat(in: -1...1) }
    var c = [Float](repeating: 0, count: n * n)
    FileHandle.standardError.write(Data("sustained: Accelerate sgemm 1024², Ctrl-C to stop\n".utf8))
    while true {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m32, m32, m32, 1, a, m32, b, m32, 0, &c, m32)
    }

case "ane":
    let batch = 32
    let sizes = InferenceEngine.layerSizes
    let dir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let url = dir.appendingPathComponent("MLPBench.mlmodelc")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    guard let model = try? MLModel(contentsOf: url, configuration: config),
          let arr = try? MLMultiArray(shape: [NSNumber(value: batch), NSNumber(value: sizes[0])], dataType: .float32) else {
        FileHandle.standardError.write(Data("ane: could not load MLPBench.mlmodelc next to the binary\n".utf8))
        exit(1)
    }
    let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: batch * sizes[0])
    for i in 0..<(batch * sizes[0]) { ptr[i] = rng.nextFloat(in: -1...1) }
    let input = try! MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: arr)])
    FileHandle.standardError.write(Data("sustained: ANE Core ML batch \(batch), Ctrl-C to stop\n".utf8))
    while true { _ = try? model.prediction(from: input) }

default:
    FileHandle.standardError.write(Data("usage: sustained sme2|neon|accelerate|ane\n".utf8))
    exit(2)
}
