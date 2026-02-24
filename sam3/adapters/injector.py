# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from fnmatch import fnmatch
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .adapter import Adapter, AdapterShapeConfig, AdapterWrapper


def _iter_parent_modules(model: nn.Module):
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            yield module, child_name, full_name, child


def _match_target(
    name: str,
    module: nn.Module,
    target_patterns: Optional[Iterable[str]],
    target_types: Optional[Iterable[str]],
) -> bool:
    pattern_ok = True
    type_ok = True
    if target_patterns:
        pattern_ok = any(fnmatch(name, pat) for pat in target_patterns)
    if target_types:
        tname = type(module).__name__
        type_ok = any(fnmatch(tname, t) for t in target_types)
    return pattern_ok and type_ok


def _infer_channel_dim(module: nn.Module, default: int = -1) -> int:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm2d)):
        return 1
    return default


def _infer_dim(module: nn.Module) -> int:
    if hasattr(module, "embed_dim"):
        return int(module.embed_dim)
    if hasattr(module, "hidden_dim"):
        return int(module.hidden_dim)
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        return int(module.num_features if hasattr(module, "num_features") else module.normalized_shape[-1])
    raise ValueError(f"Cannot infer adapter dim for module type: {type(module).__name__}")


def inject_adapters(model: nn.Module, cfg: Optional[Dict] = None) -> List[str]:
    """Inject adapters by wrapping matched child modules.

    cfg supports:
      - enabled: bool
      - target_patterns: List[str]
      - target_types: List[str]
      - bottleneck_dim: int
      - activation: str
      - dropout: float
      - layernorm_before: bool
      - init_scale: float
      - channel_dim: int
    """
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return []

    target_patterns = cfg.get("target_patterns", ["*"])
    target_types = cfg.get("target_types", ["Linear", "MultiheadAttention", "Attention", "Block"])

    wrapped = []
    for parent, child_name, full_name, child in list(_iter_parent_modules(model)):
        if isinstance(child, AdapterWrapper):
            continue
        if not _match_target(full_name, child, target_patterns, target_types):
            continue
        try:
            dim = _infer_dim(child)
        except Exception:
            continue

        adapter = Adapter(
            dim=dim,
            bottleneck_dim=int(cfg.get("bottleneck_dim", 64)),
            activation=cfg.get("activation", "gelu"),
            dropout=float(cfg.get("dropout", 0.0)),
            layernorm_before=bool(cfg.get("layernorm_before", True)),
            init_scale=float(cfg.get("init_scale", 1e-3)),
        )
        channel_dim = int(cfg.get("channel_dim", _infer_channel_dim(child, -1)))
        wrapper = AdapterWrapper(
            module=child,
            adapter=adapter,
            shape_cfg=AdapterShapeConfig(channel_dim=channel_dim),
        )
        setattr(parent, child_name, wrapper)
        wrapped.append(full_name)

    return wrapped


def is_adapter_parameter(name: str) -> bool:
    return ".adapter." in name or name.endswith(".scale") and ".adapter" in name


def freeze_except_adapters(model: nn.Module) -> Dict[str, int]:
    trainable = 0
    frozen = 0
    for n, p in model.named_parameters():
        keep = is_adapter_parameter(n)
        p.requires_grad_(keep)
        if keep:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return {"trainable": trainable, "frozen": frozen}


def get_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ".adapter." in k}


def load_adapter_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    adapter_missing = [k for k in missing if ".adapter." in k]
    adapter_unexpected = [k for k in unexpected if ".adapter." in k]
    if strict and (adapter_missing or adapter_unexpected):
        raise RuntimeError(
            f"Adapter state_dict mismatch. missing={adapter_missing}, unexpected={adapter_unexpected}"
        )
    return adapter_missing, adapter_unexpected
