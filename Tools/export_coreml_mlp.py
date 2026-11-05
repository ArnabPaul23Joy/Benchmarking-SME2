#!/usr/bin/env python3
"""Export the benchmark MLP to a Core ML .mlpackage so the app can time it on the Neural Engine.

Architecture matches DenseNetwork(sizes:) in the app: 1024->2048->2048->1024->16,
ReLU on every layer except the last. Weights are random: this model is only used to
measure Core ML / Neural Engine inference time, never for correctness.

The input has a flexible batch dimension (1...128) so one model serves every batch size
the app benchmarks. Run with a Python env that has torch + coremltools:

    python3 Tools/export_coreml_mlp.py

Output: SME2Inference/MLPBench.mlpackage (inside the file-system-synchronized group,
so Xcode compiles and bundles it automatically).
"""

import os
import torch
import torch.nn as nn
import coremltools as ct

SIZES = [1024, 2048, 2048, 1024, 16]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "SME2Inference", "MLPBench.mlpackage")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(len(SIZES) - 1):
            layers.append(nn.Linear(SIZES[i], SIZES[i + 1]))
            if i < len(SIZES) - 2:          # ReLU on all but the last layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    model = MLP().eval()
    example = torch.rand(1, SIZES[0])
    traced = torch.jit.trace(model, example)

    batch = ct.RangeDim(lower_bound=1, upper_bound=128, default=1)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(batch, SIZES[0]))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    out = os.path.normpath(OUT)
    mlmodel.save(out)
    print("Saved", out)


if __name__ == "__main__":
    main()
