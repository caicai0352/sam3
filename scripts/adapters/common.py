#!/usr/bin/env python3
# pyre-unsafe

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = ROOT / "sam3"


def import_adapters_module():
    """Import `sam3.adapters` without executing heavy `sam3/__init__.py`."""
    sam3_stub = types.ModuleType("sam3")
    sam3_stub.__path__ = [str(SAM3_ROOT)]
    sys.modules.setdefault("sam3", sam3_stub)
    from sam3 import adapters  # noqa: E402

    return adapters


class TinyRegModel(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def build_synth_data(n=128, in_dim=16, out_dim=4, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    w = torch.linspace(-0.9, 0.9, in_dim * out_dim).view(in_dim, out_dim)
    y = x @ w + 0.1 * torch.sin(x[:, :out_dim])
    return x.to(device), y.to(device)
