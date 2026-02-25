# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class AdapterShapeConfig:
    """Shape adaptation options for AdapterWrapper."""

    channel_dim: int = -1


class Adapter(nn.Module):
    """A lightweight bottleneck adapter with residual connection."""

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.0,
        layernorm_before: bool = True,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be > 0, got {bottleneck_dim}")

        self.norm = nn.LayerNorm(dim) if layernorm_before else nn.Identity()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up(h)
        return x + self.scale * h


class AdapterWrapper(nn.Module):
    """Wrap a module and apply adapter to its tensor output with shape adaption."""

    def __init__(
        self,
        module: nn.Module,
        adapter: Adapter,
        shape_cfg: Optional[AdapterShapeConfig] = None,
    ):
        super().__init__()
        self.module = module
        self.adapter = adapter
        self.shape_cfg = shape_cfg or AdapterShapeConfig()

    def _apply_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            return x

        cdim = self.shape_cfg.channel_dim
        if cdim < 0:
            cdim = x.dim() + cdim

        if cdim == x.dim() - 1:
            return self.adapter(x)

        x_perm = torch.movedim(x, cdim, -1)
        x_perm = self.adapter(x_perm)
        return torch.movedim(x_perm, -1, cdim)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return self._apply_adapter(out)
        if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            first = self._apply_adapter(out[0])
            return (first, *out[1:])
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            out = list(out)
            out[0] = self._apply_adapter(out[0])
            return out
        return out
