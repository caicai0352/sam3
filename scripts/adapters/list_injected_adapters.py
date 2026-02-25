#!/usr/bin/env python3
# pyre-unsafe

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adapters.common import TinyRegModel, import_adapters_module


def get_args():
    p = argparse.ArgumentParser(description="List injected adapter module names on a tiny model.")
    p.add_argument("--targets", nargs="+", default=["net.*"])
    p.add_argument("--module-types", nargs="+", default=["Linear"])
    p.add_argument("--d-adapter", type=int, default=8)
    return p.parse_args()


def main():
    args = get_args()
    adapters = import_adapters_module()
    model = TinyRegModel()
    adapters.inject_adapters(
        model,
        {
            "enabled": True,
            "targets": args.targets,
            "module_types": args.module_types,
            "d_adapter": args.d_adapter,
        },
    )
    names = adapters.list_injected_adapters(model)
    print(json.dumps({"count": len(names), "modules": names}, indent=2))


if __name__ == "__main__":
    main()
