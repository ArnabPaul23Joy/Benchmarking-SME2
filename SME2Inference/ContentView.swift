import SwiftUI

@MainActor
@Observable
final class AppModel {
    let device = DeviceInfo.current
    private let engine = InferenceEngine()

    var gemmSize = 1024
    var gemmReport: BenchReport?
    var batch = 32
    var nnReport: BenchReport?
    var stressSeconds = 30
    var stressReports: [StressSummary] = []
    var isRunning = false

    func runGEMM() {
        isRunning = true
        let n = gemmSize, sme = device.hasSME2
        Task {
            let report = await Task.detached(priority: .userInitiated) {
                GEMMBenchmark.run(n: n, useSME2: sme)
            }.value
            report.log(title: "GEMM benchmark · \(n)² · SME2=\(sme)")
            gemmReport = report
            isRunning = false
        }
    }

    func runInference() {
        isRunning = true
        let batch = batch, sme = device.hasSME2
        Task(priority: .userInitiated) {
            let report = await engine.run(batch: batch, useSME2: sme)
            report.log(title: "Inference · batch \(batch) · SME2=\(sme)")
            nnReport = report
            isRunning = false
        }
    }

    func runStress() {
        isRunning = true
        let seconds = Double(stressSeconds), sme = device.hasSME2
        Task {
            let reports = await Task.detached(priority: .userInitiated) {
                ThermalStressSuite.run(seconds: seconds, useSME2: sme)
            }.value
            for report in reports { report.log() }
            stressReports = reports
            isRunning = false
        }
    }
}

extension BenchReport {
    /// Prints the benchmark rows and notes to the Xcode console.
    func log(title: String) {
        print("── \(title) · thermal=\(ThermalLevel.name(ThermalLevel.current())) ──")
        for row in rows {
            print(String(format: "  %@: %.3f ms, %.0f %@", row.name, row.milliseconds, row.rate, row.unit))
        }
        for note in notes {
            print("  • \(note)")
        }
    }
}

struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        NavigationStack {
            Form {
                deviceSection
                gemmSection
                inferenceSection
                thermalSection
            }
            .navigationTitle("SME2 Inference")
            .overlay {
                if model.isRunning {
                    ProgressView("Running…")
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
        }
    }

    private var deviceSection: some View {
        Section("This device") {
            LabeledContent("Model", value: model.device.model)
            LabeledContent("FEAT_SME", value: model.device.hasSME ? "Yes" : "No")
            LabeledContent("FEAT_SME2", value: model.device.hasSME2 ? "Yes" : "No")
            if model.device.hasSME {
                LabeledContent("Streaming vector length", value: "\(model.device.streamingVectorBits)-bit")
                LabeledContent("ZA tiles (fp32)", value: "4 × \(model.device.fp32Lanes)×\(model.device.fp32Lanes)")
            }
            if !model.device.hasSME2 {
                Text("This CPU does not report SME2, so only the NEON and Accelerate paths will run.")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var gemmSection: some View {
        Section {
            Picker("Matrix size", selection: $model.gemmSize) {
                ForEach([256, 512, 1024, 2048], id: \.self) { Text(String($0) + "²").tag($0) }
            }
            Button("Run GEMM benchmark", action: model.runGEMM)
                .disabled(model.isRunning)
            if let report = model.gemmReport {
                ResultsView(report: report)
            }
        } header: {
            Text("Matrix multiply")
        } footer: {
            Text("C = A·B on one core. SME2 rows use the hand-written FMOPA/SMOPA kernels in sme_kernels.c.")
        }
    }

    private var thermalSection: some View {
        Section {
            Picker("Duration", selection: $model.stressSeconds) {
                ForEach([10, 30, 60, 120], id: \.self) { Text("\($0) s").tag($0) }
            }
            Button("Run thermal stress test", action: model.runStress)
                .disabled(model.isRunning)
            ForEach(model.stressReports) { report in
                StressResultView(report: report)
            }
        } header: {
            Text("Thermal signature (per unit)")
        } footer: {
            Text("Drives each unit at full load for the chosen time and tracks how throughput throttles as the system heats up. " +
                 "Per-unit power (watts) is not available on iOS; use Tools/measure_power.sh on an M4 Mac for real power.")
        }
    }

    private var inferenceSection: some View {
        Section {
            Picker("Batch size", selection: $model.batch) {
                ForEach([1, 8, 32, 128], id: \.self) { Text("\($0)").tag($0) }
            }
            Button("Run inference", action: model.runInference)
                .disabled(model.isRunning)
            if let report = model.nnReport {
                ResultsView(report: report)
            }
        } header: {
            Text("Neural network inference")
        } footer: {
            Text("MLP \(InferenceEngine.layerSizes.map(String.init).joined(separator: " → ")) with ReLU, random weights. " +
                 "Load real weights with DenseNetwork(contentsOf:) and Tools/export_mlp.py.")
        }
    }
}

private struct ResultsView: View {
    let report: BenchReport

    var body: some View {
        let fastest = report.rows.map(\.rate).max() ?? 1
        VStack(alignment: .leading, spacing: 10) {
            ForEach(report.rows) { row in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(row.name).font(.subheadline.weight(.medium))
                        Spacer()
                        Text(String(format: "%.3f ms", row.milliseconds)).monospacedDigit()
                    }
                    HStack(spacing: 8) {
                        GeometryReader { geo in
                            Capsule()
                                .fill(row.name.hasPrefix("SME2") ? Color.accentColor : Color.secondary.opacity(0.5))
                                .frame(width: max(4, geo.size.width * row.rate / fastest))
                        }
                        .frame(height: 6)
                        Text(String(format: "%.0f %@", row.rate, row.unit))
                            .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                            .frame(minWidth: 90, alignment: .trailing)
                    }
                }
            }
            ForEach(report.notes, id: \.self) { note in
                Text(note).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

private struct StressResultView: View {
    let report: StressSummary

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(report.unit).font(.subheadline.weight(.medium))
            if report.available {
                HStack {
                    Text(String(format: "peak %.0f → end %.0f GFLOPS", report.peakGFLOPS, report.endGFLOPS))
                        .monospacedDigit()
                    Spacer()
                    Text(String(format: "%.0f%% throttle", report.throttlePercent))
                        .foregroundStyle(report.throttlePercent > 5 ? .orange : .secondary)
                        .monospacedDigit()
                }
                .font(.caption)
                Text("thermal \(ThermalLevel.name(report.startThermal)) → \(ThermalLevel.name(report.maxThermal))")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text(report.note).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    ContentView()
}
