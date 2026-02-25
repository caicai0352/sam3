# SAM3 Adapter 使用说明（训练 / 评估 / 可视化）

> 本文档所有命令都对应仓库内真实脚本与参数，可直接复制执行。

---

## A. 目标与能力

### A.1 适配范围
当前 adapter 注入能力支持按模块名/类型匹配注入，可用于以下关键区域（由 `adapter_cfg.targets/module_types` 决定）：
- **text backbone**（如 `VETextEncoder / TextTransformer / ResidualAttentionBlock`）
- **det head**（`TransformerDecoder` 相关子模块）
- **seg head**（`SegmentationHead / UniversalSegmentationHead / PixelDecoder / MaskPredictor`）
- **transformer / ViT block**（如 `vitdet.Block`、`sam.transformer.TwoWayAttentionBlock`）

实现入口：
- `sam3/adapters/adapter.py`
- `sam3/adapters/injector.py`
- 模型构建接入：`sam3/model_builder.py`

### A.2 训练模式
支持 **冻结主干，仅训练 adapter**，并可选放开 LN/bias：
- `freeze_except_adapters=True`
- `train_ln=True/False`
- `train_bias=True/False`

相关实现：
- `sam3/adapters/injector.py::freeze_except_adapters`
- `sam3/train/trainer.py::OptimConf`

### A.3 checkpoint 策略
- **full checkpoint**：保存全部模型参数
- **adapter-only checkpoint**：仅保存 `".adapter."` 参数

训练恢复时可按需加载 adapter-only：
- `save_adapter_only=True`
- `load_adapter_only=True`

相关实现：
- `sam3/train/trainer.py`
- `sam3/adapters/injector.py::{get_adapter_state_dict, load_adapter_state_dict}`

---

## B. 快速开始（复制即可跑）

### B.1 环境安装（conda / pip 二选一）

#### 方案1：conda
```bash
conda create -n sam3 python=3.10 -y
conda activate sam3
pip install -e ".[train,dev]"
```

#### 方案2：pip（当前环境）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

> 依赖来源：项目根目录 `pyproject.toml`（extras: `train`, `dev`）。

### B.2 数据准备

#### 最小 smoke（推荐先跑）
无需外部数据。`scripts/adapters/*` 使用合成数据（见 `scripts/adapters/common.py`）。

#### 真实训练
使用仓库已有训练配置（例如 `sam3/train/configs/odinw13/odinw_text_only_train.yaml`），按配置中的数据路径准备目录。

### B.3 1 行命令跑通 smoke test
```bash
python scripts/adapter_smoke_test.py --output-dir outputs/adapter_smoke/full --steps 20 --device cpu
```

### B.4 1 行命令跑通训练（小步数）
```bash
python scripts/adapters/train_adapter_smoke.py --output-dir outputs/adapter_smoke/adapter --use-adapter --freeze-base --save-adapter-only --steps 40 --device cpu
```

### B.5 1 行命令跑通评估
```bash
python scripts/adapters/eval_adapter_smoke.py --base-ckpt outputs/adapter_smoke/adapter/base_or_full.pt --adapter-ckpt outputs/adapter_smoke/adapter/adapter_only.pt --use-adapter --output outputs/adapter_smoke/adapter/metrics.json --device cpu
```

### B.6 1 行命令跑通可视化
```bash
python scripts/adapters/visualize_adapter_compare.py --base-ckpt outputs/adapter_smoke/base/base_or_full.pt --adapter-base-ckpt outputs/adapter_smoke/adapter/base_or_full.pt --adapter-ckpt outputs/adapter_smoke/adapter/adapter_only.pt --output-image outputs/adapter_smoke/compare_base_vs_adapter.png --device cpu
```

### B.7 NVIDIA A800（GPU）
```bash
python scripts/adapter_smoke_test.py --output-dir outputs/adapter_smoke/a800 --steps 40 --device cuda
```

---

## C. 配置说明

### C.1 adapter 配置字段（`inject_adapters(model, adapter_cfg)`）
支持字段（含别名）：
- `enabled`: 是否启用
- `target_patterns` / `targets`: 模块名匹配（支持通配符）
- `target_types` / `module_types`: 模块类型名匹配（如 `Linear`）
- `bottleneck_dim` / `d_adapter`: adapter bottleneck 维度
- `dropout`: adapter dropout
- `init_scale` / `init`: adapter 残差缩放初值
- `activation`: `gelu/relu/silu`
- `layernorm_before`: adapter 内部是否 pre-LN
- `channel_dim`: 非最后维通道时用于 shape 适配
- `placement`: 当前支持 `post/output`
- `houlsby`: `True` 时按后置残差路径处理（等价 post）

### C.2 冻结策略（训练时）
- `freeze_except_adapters=True`：只训练 adapter
- `train_ln=True`：额外训练 LN 参数
- `train_bias=True`：额外训练 bias 参数

### C.3 optimizer param groups 与 LR 建议
- 基础建议：`adapter_lr_scale=1.0`
- 常见 PEFT 设置：`adapter_lr_scale=5~10`（adapter 通常可用更高 LR）
- 当 `freeze_except_adapters=True` 时，trainer 会优先构造 adapter 参数组。

---

## D. 常见问题（FAQ）

### D.1 target 匹配不到怎么办？
先列出实际已注入模块：
```bash
python scripts/adapters/list_injected_adapters.py --targets net.* --module-types Linear --d-adapter 8
```
若为空：
1) 放宽 `targets`（如 `*`）
2) 检查 `module_types` 是否与真实类名一致
3) 先在小模型上验证再迁移到 SAM3

### D.2 只加载 adapter 报 shape mismatch 怎么排查？
重点核对：
1) `d_adapter/bottleneck_dim` 是否一致
2) `target_patterns/module_types` 是否一致
3) 基座模型结构与版本是否一致
4) 先用 `strict=False` 定位 missing/unexpected，再修正配置

### D.3 关闭 adapter 后输出不一致如何定位？
建议顺序：
1) 设定固定随机种子
2) 确认未加载 adapter-only 权重
3) 确认 `enabled=False` 且模型中未注入 adapter
4) 对比同一输入下中间层输出（先从注入层开始）

---

## 相关脚本与文档
- `docs/ADAPTER_WORKFLOW.md`
- `scripts/adapter_smoke_test.py`
- `scripts/adapters/train_adapter_smoke.py`
- `scripts/adapters/eval_adapter_smoke.py`
- `scripts/adapters/visualize_adapter_compare.py`
- `scripts/adapters/list_injected_adapters.py`
- `tests/test_adapter_smoke.py`
