# Task 2: 训练效率优化

## 1. 背景与任务目标

**任务**: 分析 `runs/speedrun.sh` 训练 pipeline，提出 2-3 个具体优化建议，实现至少 1 个，并提供 before/after 对比数据。

**解决的问题**: nanochat 当前 d12 单卡 H100 baseline 约 960ms/step、546K tok/sec、42% MFU。GPU 利用率仍有提升空间，优化训练吞吐可以直接缩短实验迭代周期。

**为什么值得做**: 训练效率是 LLM 开发的核心瓶颈之一。即使 5% 的提升，在大规模训练中也意味着数小时甚至数天的节省。

---

## 2. 现有架构理解

### `runs/speedrun.sh` Pipeline 分析

Pipeline 包含四个阶段：

1. **数据准备**: 下载 8 个 shard (~800MB)，后台继续下载 170 个 shard；训练 BPE tokenizer (vocab=32768)；评估压缩率。
2. **Base model 预训练**: `torchrun --nproc_per_node=8` 训练 d24 模型，使用 `--target-param-data-ratio=8`、`--fp8`、`--device-batch-size=16`。这是主要的计算阶段（8×H100 约 2.5 小时）。
3. **SFT**: 基于对话数据 + identity 数据微调 base model。
4. **评估与报告**: CORE metric、BPB、采样、生成 markdown 报告。

### 训练循环瓶颈分析

每一步训练包含以下计算：

| 组件 | 说明 | 开销 |
|------|------|------|
| Forward + Backward | GPU-bound，受 matmul 吞吐和显存带宽限制 | ~85-90% |
| Muon 优化器 (Polar Express) | 5 次 PE 迭代 × 3 shape 组 × 3 matmul = 45 次 batched matmul | ~5-10% |
| Python/CUDA 调度开销 | kernel launch latency、Python 解释器开销、CPU-GPU 同步 | ~3-5% |

### 涉及的原始模块

| 文件 | 职责 |
|------|------|
| `scripts/base_train.py` | 训练主循环：CLI 参数、模型初始化、数据加载、优化器配置、训练/评估 |
| `nanochat/optim.py` | MuonAdamW 优化器实现，Polar Express 正交化（`ns_steps` 参数） |
| `nanochat/gpt.py` | GPT 模型定义，`setup_optimizer()` 初始化优化器参数组 |
| `runs/speedrun.sh` | 端到端训练 pipeline 脚本 |

---

## 3. 设计方案

提出三个优化建议，实现前两个。

### 优化 1: `torch.compile mode="reduce-overhead"`（已实现）

**原理**: 默认 `torch.compile` 模式融合 kernel 但仍逐个从 Python 启动。`mode="reduce-overhead"` 启用 CUDA graph 捕获——将整个 forward+backward 录制为 CUDA graph，后续步骤直接 replay，消除：
- Python 解释器在 op 之间的开销
- CUDA kernel 启动延迟 (~5-10μs/kernel × 数百个 kernel)
- CPU-GPU 同步点

**前置条件**（nanochat 均已满足）：
- 固定 tensor shape (`dynamic=False`)
- 固定 batch size 和 sequence length
- 数据预取在 compiled 区域外部

**限制**: 与 `--fp8` 互斥（FP8 使用动态 amax 缩放，每步变化，与 graph replay 不兼容）。

**预期收益**: ~3-8%。

### 优化 2: Polar Express `ns_steps` 5→3 调度（已实现）

**原理**: Muon 优化器使用 Polar Express (Newton-Schulz 迭代) 近似矩阵正交化。默认 `ns_steps=5`，即每步优化器执行 45 次 batched matmul（5 迭代 × 3 shape 组 × 3 matmul/迭代）。

Warmup 后梯度趋于平滑，正交化精度可以降低。从 5 降到 3 省去 40% 的 PE 计算（每步减少 18 次 matmul）。

**调度策略**: 前 80 步 `ns_steps=5`（warmup），之后 `ns_steps=3`。切换点 step 80 < 统计窗口起点 step 100，重编译不影响性能统计。

**预期收益**: ~3-5%。

### 优化 3: FP8 训练（仅分析——已有实现）

**现状**: nanochat 已通过 `--fp8` flag 实现 FP8 训练，支持 tensorwise/rowwise scaling。

**已有数据**（LOG.md，d26 8×H100）：
- 原始吞吐提升 1.17×
- 质量匹配后净收益 ~5%

**d12 分析**: d12 规模较小，更偏 memory-bandwidth-bound 而非 compute-bound。FP8 的优势主要来自 H100 tensor core matmul 吞吐翻倍，但 d12 的 matmul 尺寸较小，不太可能 compute-saturated。预期净收益 0-3%。

**限制**: 与 `mode="reduce-overhead"` 互斥（动态 amax 缩放）。

### 备选方案对比

| 方案 | 实现复杂度 | 预期收益 | 质量风险 | 与 FP8 兼容 |
|------|:-:|:-:|:-:|:-:|
| reduce-overhead | 低 | 3-8% | 无 | 否 |
| ns_steps 调度 | 低 | 3-5% | 低 | 是 |
| FP8 (已有) | — | 0-3% (d12) | 低 | — |

**Trade-off**: reduce-overhead 和 FP8 互斥，需要根据模型规模选择。d12 上 reduce-overhead 预期优于 FP8；d24+ 上 FP8 更有价值。ns_steps 调度与两者均兼容，可以叠加。

---

## 4. 关键实现说明

### 修改文件

仅修改 `scripts/base_train.py`，未修改任何其他文件。

### 新增 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子，用于实验可复现 |
| `--compile-mode` | `default` | `torch.compile` 模式：`default`/`reduce-overhead`/`max-autotune` |
| `--ns-steps` | 5 | Muon 优化器 Polar Express 初始迭代次数 |
| `--ns-steps-final` | -1 | warmup 后降低到的 ns_steps（-1 = 不调度） |
| `--ns-steps-warmup` | 80 | 降低 ns_steps 前的 warmup 步数 |

所有新参数均有合理默认值，**不影响原有默认行为**（向后兼容）。

### 关键实现点

**1. Seed 初始化**（`compute_init` 之后）
```python
import random
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```

**2. FP8 + reduce-overhead 互斥校验**（FP8 初始化之前）
```python
if args.fp8 and args.compile_mode == "reduce-overhead":
    raise ValueError("--fp8 and --compile-mode reduce-overhead are mutually exclusive: "
                     "FP8 dynamic amax scaling is incompatible with CUDA graph capture")
```

**3. `torch.compile` 分支**
```python
if args.compile_mode == "default":
    model = torch.compile(model, dynamic=False)
else:
    model = torch.compile(model, dynamic=False, mode=args.compile_mode)
```

**4. ns_steps 调度函数**
```python
def get_ns_steps(it):
    if args.ns_steps_final == -1 or args.ns_steps_final == args.ns_steps:
        return args.ns_steps
    return args.ns_steps if it < args.ns_steps_warmup else args.ns_steps_final
```

**5. 训练循环中每步更新 ns_steps**（与 LR/momentum/weight_decay 一同更新）
```python
ns_steps = get_ns_steps(step)
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group['kind'] == 'muon':
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
        group["ns_steps"] = ns_steps  # 新增
```

### 容易出错的点

- `setup_optimizer()` 中硬编码 `ns_steps=5`，需要在优化器创建后用 CLI 参数覆盖
- ns_steps 切换触发 `torch.compile` 重编译，必须在统计窗口之前完成（step 80 < step 100）
- FP8 的动态 amax 缩放与 CUDA graph 不兼容，必须显式拦截

---

## 5. 验证方法

### 性能验证

- **统计口径**: step ≥ 100（排除编译 warmup 和 ns_steps 切换重编译）
- **指标**: ms/step、tok/sec、MFU，取 step 100-999 的均值和标准差
- **可复现性**: baseline 和最优配置各跑 2 个 seed（42, 137）

### 质量验证

- **指标**: val bpb @250/500/750/1000
- **退化阈值**: ≤ 0.005（超过则判定不可用）

### 提升计算公式

- **吞吐提升** = (tok/sec_variant − tok/sec_base) / tok/sec_base × 100%
- **时延改善** = (ms_base − ms_variant) / ms_base × 100%

---

## 6. 结果展示

### 实验配置

| # | 配置 | 额外 CLI 参数 | Seed |
|---|------|--------------|------|
| 1a | Baseline | — | 42 |
| 1b | Baseline (repeat) | — | 137 |
| 2 | reduce-overhead | `--compile-mode reduce-overhead` | 42 |
| 3 | ns_steps 5→3 | `--ns-steps 5 --ns-steps-final 3 --ns-steps-warmup 80` | 42 |
| 4 | 组合 | reduce-overhead + ns_steps 5→3 | 42 |
| 5 | 最优 (repeat) | （跑完 2-4 后确定） | 137 |

### 性能对比（step ≥ 100）

| 配置 | ms/step avg±std | tok/sec avg±std | MFU% | 总训练时间 | val bpb @1000 | bpb 退化 | 吞吐提升 |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Baseline (seed 42) | — | — | — | — | — | — | — |
| Baseline (seed 137) | — | — | — | — | — | — | — |
| reduce-overhead | — | — | — | — | — | — | — |
| ns_steps 5→3 | — | — | — | — | — | — | — |
| 组合 | — | — | — | — | — | — | — |
| 最优 (seed 137) | — | — | — | — | — | — | — |

### 质量对比（val bpb）

| 配置 | @250 | @500 | @750 | @1000 |
|------|:----:|:----:|:----:|:-----:|
| Baseline (seed 42) | — | — | — | — |
| Baseline (seed 137) | — | — | — | — |
| reduce-overhead | — | — | — | — |
| ns_steps 5→3 | — | — | — | — |
| 组合 | — | — | — | — |
| 最优 (seed 137) | — | — | — | — |

*（待实验完成后填入）*

---

## 7. 局限性

1. **reduce-overhead 与 FP8 互斥**: 无法同时使用，大模型（d24+）上需要在两者之间选择
2. **ns_steps=3 的质量影响未经大规模验证**: 1000 步实验可能不足以暴露长期训练中的质量退化
3. **单卡实验**: 优化效果可能在多卡 DDP 场景下有所不同（通信开销占比变化）
4. **统计噪声**: 1000 步训练的 ms/step 波动较大，通过 2 seed 缓解但非完全消除
5. **ns_steps 调度策略简单**: 仅使用阶梯式切换，未探索渐进式降低或自适应策略

### 后续可改进方向

- 探索 `ns_steps` 自适应调度（根据梯度范数动态调整）
- 在 d24/d26 规模上验证优化效果
- 测试 `max-autotune` 模式（编译时间更长但可能找到更优 kernel）
- 结合 FP8 + ns_steps 调度（两者兼容）在大模型上的组合效果

---

## 8. 运行说明

### 环境依赖

- 硬件: NVIDIA H100 SXM GPU
- 软件: PyTorch 2.x, nanochat 依赖 (`uv sync --extra gpu`)
- 数据: 预训练数据集已下载 (`python -m nanochat.dataset -n 170`)

### 运行命令

```bash
# 激活环境
source .venv/bin/activate && export NANOCHAT_BASE_DIR=/workspace/nanochat_data

# 公共参数
BASE="--depth=12 --num-iterations 1000 --eval-every 250 --save-every -1 --core-metric-every -1 --sample-every -1"

# 1a. Baseline (seed 42)
python -m scripts.base_train $BASE --model-tag d12_opt_base   --seed 42

# 1b. Baseline (seed 137)
python -m scripts.base_train $BASE --model-tag d12_opt_base2  --seed 137

# 2. reduce-overhead
python -m scripts.base_train $BASE --model-tag d12_opt_ro     --seed 42  --compile-mode reduce-overhead

# 3. ns_steps 5→3
python -m scripts.base_train $BASE --model-tag d12_opt_ns     --seed 42  --ns-steps 5 --ns-steps-final 3 --ns-steps-warmup 80

# 4. 组合
python -m scripts.base_train $BASE --model-tag d12_opt_both   --seed 42  --compile-mode reduce-overhead --ns-steps 5 --ns-steps-final 3 --ns-steps-warmup 80

# 5. 最优配置 repeat（待确定配置后追加）
```

### 日志收集

训练日志保存在 `dev/logs/task2_*.log`，使用 `tee` 或 screen 的 `-L` 参数捕获。

### 快速验证（面试官）

```bash
# 最小化验证：运行 20 步，确认新参数生效
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --eval-tokens=512 --total-batch-size=512 --num-iterations=20 \
    --compile-mode reduce-overhead --ns-steps 5 --ns-steps-final 3 --ns-steps-warmup 10 \
    --seed 42 --save-every -1 --core-metric-every -1 --sample-every -1
```

---

## 9. 修改文件说明

| 文件 | 修改内容 | 为什么改 | 是否影响默认行为 |
|------|---------|---------|:---:|
| `scripts/base_train.py` | +5 CLI 参数、seed 初始化、FP8 互斥校验、compile 分支、ns_steps 调度函数、训练循环 ns_steps 更新 | 训练效率优化的所有逻辑均在训练脚本中 | 否（所有新参数有默认值，与原行为一致） |
| `README_training_efficiency.md` | 新建 | 交付文档 | 否 |
