# SAM3 Adapter Workflow (Train / Eval / Visualization)

## 1) Repo status scan (existing vs gaps)

### Existing
- **Training entrypoint**: `sam3/train/train.py` (Hydra + Trainer).  
- **Trainer/Checkpoint infra**: `sam3/train/trainer.py` + `sam3/train/utils/checkpoint_utils.py`.  
- **Optimizer infra**: `sam3/train/optim/optimizer.py`.  
- **Model build entrypoints**: `sam3/model_builder.py` (`build_sam3_image_model`, `build_sam3_video_model`).  
- **Evaluation ecosystem**: `scripts/eval/*` + `sam3/eval/*` and eval configs in `sam3/train/configs/*eval*`.  
- **Adapter v1 core**: `sam3/adapters/adapter.py`, `sam3/adapters/injector.py`.

### Gaps (before this patch)
- No **single README** for adapter end-to-end flow with reproducible commands.
- No CPU-friendly scripts covering **adapter on/off**, **base+adapter composition loading**, **adapter-only save/load**, **metrics.json**, **visual compare image**.
- Existing `scripts/adapter_smoke_test.py` had no argparse CLI help and did not define a full train/eval/vis pipeline.

### File-level implementation plan
- **Add** `scripts/adapters/common.py` (shared tiny model/data + safe adapter import)
- **Add** `scripts/adapters/train_adapter_smoke.py` (train + checkpoint save, adapter on/off)
- **Add** `scripts/adapters/eval_adapter_smoke.py` (load and evaluate -> `metrics.json`)
- **Add** `scripts/adapters/visualize_adapter_compare.py` (base vs adapter compare png)
- **Add** `tests/test_adapter_smoke.py` (unittest smoke)
- **Update** `scripts/adapter_smoke_test.py` (argparse `--help`, call pipeline)
- **Add** `docs/ADAPTER_WORKFLOW.md` (this doc)

---

## 2) Reproducible commands

> All commands below map to real scripts and support `--help`.

### 2.1 Adapter OFF baseline train (CPU smoke)
```bash
python scripts/adapters/train_adapter_smoke.py \
  --output-dir outputs/adapter_smoke/base \
  --steps 40
```

### 2.2 Adapter ON train + freeze base + save adapter-only
```bash
python scripts/adapters/train_adapter_smoke.py \
  --output-dir outputs/adapter_smoke/adapter \
  --use-adapter \
  --freeze-base \
  --save-adapter-only \
  --steps 40
```

### 2.3 Evaluate baseline -> metrics.json
```bash
python scripts/adapters/eval_adapter_smoke.py \
  --base-ckpt outputs/adapter_smoke/base/base_or_full.pt \
  --output outputs/adapter_smoke/base/metrics.json
```

### 2.4 Evaluate adapter model (base+adapter composed load) -> metrics.json
```bash
python scripts/adapters/eval_adapter_smoke.py \
  --base-ckpt outputs/adapter_smoke/adapter/base_or_full.pt \
  --adapter-ckpt outputs/adapter_smoke/adapter/adapter_only.pt \
  --use-adapter \
  --output outputs/adapter_smoke/adapter/metrics.json
```

### 2.5 Visualization compare (base vs adapter)
```bash
python scripts/adapters/visualize_adapter_compare.py \
  --base-ckpt outputs/adapter_smoke/base/base_or_full.pt \
  --adapter-base-ckpt outputs/adapter_smoke/adapter/base_or_full.pt \
  --adapter-ckpt outputs/adapter_smoke/adapter/adapter_only.pt \
  --output-image outputs/adapter_smoke/compare_base_vs_adapter.png
```

### 2.6 One-command smoke test wrapper
```bash
python scripts/adapter_smoke_test.py --output-dir outputs/adapter_smoke/full
```

### 2.7 Unit test
```bash
python -m unittest tests/test_adapter_smoke.py
```

---

## 3) Output artifacts

- Train report: `outputs/.../train_report.json`
- Eval metrics: `outputs/.../metrics.json`
- Compare image: `outputs/.../compare_base_vs_adapter.png`

These artifacts are intended for quick CI/local verification on CPU.
