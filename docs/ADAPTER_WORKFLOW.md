# SAM3 Adapter 工作流（训练 / 评估 / 可视化）

## 1）仓库现状扫描（已有能力 vs 缺口）

### 已有能力
- **训练入口**：`sam3/train/train.py`（Hydra + Trainer）
- **训练器与断点机制**：`sam3/train/trainer.py` + `sam3/train/utils/checkpoint_utils.py`
- **优化器机制**：`sam3/train/optim/optimizer.py`
- **模型构建入口**：`sam3/model_builder.py`（`build_sam3_image_model`、`build_sam3_video_model`）
- **评估生态**：`scripts/eval/*` + `sam3/eval/*`，以及 `sam3/train/configs/*eval*`
- **Adapter 初版能力**：`sam3/adapters/adapter.py`、`sam3/adapters/injector.py`

### 之前的缺口（本次补齐前）
- 缺少一个 **单一中文文档** 来描述 adapter 端到端复现流程
- 缺少 CPU 友好的标准脚本，覆盖：
  - adapter on/off
  - base + adapter 组合加载
  - adapter-only 保存/加载
  - `metrics.json` 输出
  - 对比可视化图片输出
- 原 `scripts/adapter_smoke_test.py` 缺少完整流程编排和清晰 CLI 说明

### 文件级改动计划（已落地）
- 新增 `scripts/adapters/common.py`（共享 tiny 模型/数据 + 轻量 adapter 导入）
- 新增 `scripts/adapters/train_adapter_smoke.py`（训练 + checkpoint 保存，支持 adapter on/off）
- 新增 `scripts/adapters/eval_adapter_smoke.py`（评估并输出 `metrics.json`）
- 新增 `scripts/adapters/visualize_adapter_compare.py`（base vs adapter 可视化）
- 新增 `tests/test_adapter_smoke.py`（单测 smoke）
- 更新 `scripts/adapter_smoke_test.py`（统一入口，带 `--help`）

---

## 2）可复现命令（全部为真实脚本）

> 下述命令均可直接运行，且脚本均支持 `--help`。

### 2.1 训练：Adapter 关闭（baseline，CPU smoke）
```bash
python scripts/adapters/train_adapter_smoke.py \
  --output-dir outputs/adapter_smoke/base \
  --steps 40
```

### 2.2 训练：Adapter 开启 + 冻结基座 + 保存 adapter-only
```bash
python scripts/adapters/train_adapter_smoke.py \
  --output-dir outputs/adapter_smoke/adapter \
  --use-adapter \
  --freeze-base \
  --save-adapter-only \
  --steps 40
```

### 2.3 评估：baseline，输出 `metrics.json`
```bash
python scripts/adapters/eval_adapter_smoke.py \
  --base-ckpt outputs/adapter_smoke/base/base_or_full.pt \
  --output outputs/adapter_smoke/base/metrics.json
```

### 2.4 评估：base + adapter 组合加载，输出 `metrics.json`
```bash
python scripts/adapters/eval_adapter_smoke.py \
  --base-ckpt outputs/adapter_smoke/adapter/base_or_full.pt \
  --adapter-ckpt outputs/adapter_smoke/adapter/adapter_only.pt \
  --use-adapter \
  --output outputs/adapter_smoke/adapter/metrics.json
```

### 2.5 可视化：base vs adapter 对比图
```bash
python scripts/adapters/visualize_adapter_compare.py \
  --base-ckpt outputs/adapter_smoke/base/base_or_full.pt \
  --adapter-base-ckpt outputs/adapter_smoke/adapter/base_or_full.pt \
  --adapter-ckpt outputs/adapter_smoke/adapter/adapter_only.pt \
  --output-image outputs/adapter_smoke/compare_base_vs_adapter.png
```

### 2.6 一键全流程 smoke（训练 + 评估 + 可视化）
```bash
python scripts/adapter_smoke_test.py --output-dir outputs/adapter_smoke/full
```

### 2.7 单元测试
```bash
python -m unittest tests/test_adapter_smoke.py
```

---

## 3）产物说明

- 训练报告：`outputs/.../train_report.json`
- 评估指标：`outputs/.../metrics.json`
- 对比可视化：`outputs/.../compare_base_vs_adapter.png`

以上产物用于本地/CI 的快速可复现验证（默认 CPU）。
