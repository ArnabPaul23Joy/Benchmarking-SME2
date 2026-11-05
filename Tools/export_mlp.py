"""Export a PyTorch MLP (nn.Linear + nn.ReLU layers) for DenseNetwork(contentsOf:).

Usage:
    python export_mlp.py model.pt weights.smew

File format (little-endian):
    uint32 magic = 0x57454D53 ("SMEW")
    uint32 layer_count
    per layer:
        uint32 in_features, uint32 out_features, uint32 relu (0/1)
        float32[in_features * out_features]   weights, row-major [in][out] (= Linear.weight.T)
        float32[out_features]                 bias
"""
import struct
import sys

import torch
import torch.nn as nn


def export(model: nn.Module, path: str) -> None:
    modules = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.ReLU))]
    layers = []
    for i, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            relu = i + 1 < len(modules) and isinstance(modules[i + 1], nn.ReLU)
            layers.append((m, relu))
    with open(path, "wb") as f:
        f.write(struct.pack("<II", 0x57454D53, len(layers)))
        for linear, relu in layers:
            w = linear.weight.detach().float().t().contiguous()  # [in, out]
            b = linear.bias.detach().float() if linear.bias is not None else torch.zeros(linear.out_features)
            f.write(struct.pack("<III", linear.in_features, linear.out_features, int(relu)))
            f.write(w.numpy().astype("<f4").tobytes())
            f.write(b.numpy().astype("<f4").tobytes())
    print(f"wrote {len(layers)} layers to {path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    export(torch.load(sys.argv[1], weights_only=False), sys.argv[2])
