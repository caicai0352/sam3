# pyre-unsafe

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.adapters.common import TinyRegModel, import_adapters_module


class AdapterSmokeTest(unittest.TestCase):
    def test_inject_and_adapter_state_roundtrip(self):
        adapters = import_adapters_module()
        m1 = TinyRegModel()
        wrapped = adapters.inject_adapters(
            m1,
            {
                "enabled": True,
                "target_patterns": ["net.0", "net.2", "net.4"],
                "target_types": ["Linear"],
                "bottleneck_dim": 8,
            },
        )
        self.assertEqual(len(wrapped), 3)

        adapters.freeze_except_adapters(m1)
        x = torch.randn(4, 16)
        y = m1(x)
        self.assertEqual(tuple(y.shape), (4, 4))

        ad_sd = adapters.get_adapter_state_dict(m1)
        self.assertTrue(len(ad_sd) > 0)

        m2 = TinyRegModel()
        adapters.inject_adapters(
            m2,
            {
                "enabled": True,
                "target_patterns": ["net.0", "net.2", "net.4"],
                "target_types": ["Linear"],
                "bottleneck_dim": 8,
            },
        )
        missing, unexpected = adapters.load_adapter_state_dict(m2, ad_sd)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ad.pt"
            torch.save({"model": ad_sd, "adapter_only": True}, p)
            loaded = torch.load(p, map_location="cpu")
            self.assertTrue(loaded["adapter_only"])


if __name__ == "__main__":
    unittest.main()
