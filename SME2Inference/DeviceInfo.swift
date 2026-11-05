import Foundation

/// What the CPU reports about its matrix extensions.
struct DeviceInfo: Sendable {
    let model: String
    let hasSME: Bool
    let hasSME2: Bool
    let streamingVectorBits: Int

    static let current: DeviceInfo = {
        var buf = [CChar](repeating: 0, count: 64)
        sme_device_model(&buf, buf.count)
        let model = buf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
        return DeviceInfo(
            model: model,
            hasSME: sme_cpu_has_sme(),
            hasSME2: sme_cpu_has_sme2(),
            streamingVectorBits: Int(sme_streaming_vector_bytes()) * 8
        )
    }()

    /// fp32 lanes per streaming vector; ZA holds four tiles of this size squared.
    var fp32Lanes: Int { streamingVectorBits / 32 }
}
