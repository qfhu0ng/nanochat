# Task 3: 模型架构变体验证 — 交付文档

## 1. 任务目标

在 nanochat d12 baseline 基础上，实现至少一种架构改进，保持参数量基本一致，训练至少 1000 步并记录 loss 曲线。我们实现了三种变体并完成了 5 组对比实验。

## 2. 架构选择与设计理由

### 2.1 SwiGLU MLP（变体 S）

**选择理由**：SwiGLU 是 LLaMA/Mistral/Gemma 等主流大模型的标准 MLP 组件，用门控机制 `SiLU(xW_gate) ⊙ xW_up` 替换 nanochat 的 `ReLU²(xW)` 激活。理论上门控激活能更好地控制信息流，提升表达能力。

**预期效果**：在文献中 SwiGLU 通常优于 vanilla ReLU/GELU。但 nanochat baseline 用的是 ReLU²（平方操作本身有类 gating 效果），因此预期差距可能较小。

**参数量设计**：精确等参。`hidden_dim = round_up(8/3 × n_embd, 128) = 2048`，使得 `3 × 768 × 2048 = 2 × 768 × 3072 = 4,718,592`（每层 MLP 参数完全相等）。

```
SwiGLU(x) = W_down @ (SiLU(W_gate @ x) ⊙ (W_up @ x))
```

### 2.2 Multi-head Latent Attention / MLA（变体 M）

**选择理由**：MLA 源自 DeepSeek-V2，通过 KV 低秩压缩大幅减少注意力参数和 KV cache 大小。选择 MLA 是为了验证"用更少参数达到相近性能"的参数效率假设。

**预期效果**：注意力参数减少 17%，总参数减少 1.6%。在参数更少的条件下，如果 bpb 接近 baseline，则说明 MLA 的参数效率更高。

**关键设计**：
- KV 低秩压缩：`x → W_kv_down(n_embd→256) → W_k_up/W_v_up` 减少参数
- 解耦 RoPE：单独的 `W_k_rope` 绕过压缩层，保留位置编码精度
- 缓存展开后的 K,V（不缓存 latent），保持 Flash Attention 3 + KVCache 完全兼容

```
latent = W_kv_down @ x          # 低秩压缩: n_embd → kv_lora_rank
K_content = W_k_up @ latent     # 展开: kv_lora_rank → n_kv_head × content_dim
V = W_v_up @ latent             # 展开: kv_lora_rank → n_kv_head × head_dim
K_rope = W_k_rope @ x           # 解耦 RoPE: 绕过压缩
K = cat(K_content, K_rope)       # 拼接后 shape 与 baseline 一致
```

### 2.3 Learnable RMSNorm（变体 N）

**选择理由**：nanochat 的 RMSNorm 是无参数的（只做归一化，无可学习缩放）。添加可学习 gamma 能让模型自适应调节每个特征维度的缩放，理论上增加表达能力，且参数开销极小。

**预期效果**：+0.007% 参数（可忽略），可能带来微小但正向的改善。作为轻量级改动，主要验证"无参数 norm 是否已经足够"。

**应用位置**：嵌入后 norm、每层 attention 前 norm、每层 MLP 前 norm、最终 lm_head 前 norm（共 2 + 12×2 = 26 个 gamma 向量）。

```
LearnableRMSNorm(x) = RMSNorm(x) × γ    # γ ∈ R^d, init = 1.0
```

## 3. 实现要点

### 3.1 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `nanochat/gpt.py` | GPTConfig +5 字段；SwiGLUMLP/MLAttention/LearnableRMSNorm 新类；Block/GPT 条件分支；init_weights 按 isinstance 分支；setup_optimizer ndim 过滤；estimate_flops/num_scaling_params 更新 |
| `nanochat/checkpoint_manager.py` | `_patch_missing_config_keys` 新增 5 字段默认值 |
| `scripts/base_train.py` | +5 CLI 参数（`--mlp-variant`, `--attn-variant`, `--use-learnable-norm`, `--kv-lora-rank`, `--rope-head-dim`）；GPTConfig 传参；auto model-tag 命名 |
| `tests/test_arch_variants.py` | 新增 24 个测试 |

### 3.2 关键设计决策

1. **向后兼容**：默认 config（relu2 + standard + no learnable norm）时 state_dict keys 与旧模型完全一致，`strict=True` 正常通过
2. **Optimizer ndim 过滤**：LearnableRMSNorm 的 gamma 是 1D 参数，必须走 AdamW（Muon 假设参数为 2D 矩阵，1D 会 crash）。用 `p.ndim >= 2` 统一分流
3. **非法值校验**：`attn_variant`/`mlp_variant` 传入非法值时 assert 报错，避免静默回退
4. **MLA 参数效率定位**：不宣称等参，而是作为"参数效率变体"——用更少参数探索性能边界

### 3.3 遇到的问题与解决

**问题 1: LearnableRMSNorm gamma 未初始化导致模型完全不学习**

nanochat 在 `torch.device("meta")` 上构建模型，`__init__` 中的 `nn.Parameter(torch.ones(dim))` 只创建无数据的 meta tensor。`to_empty()` 后 gamma 存储有了但内容是垃圾值，而 `init_weights()` 没有重置它。结果：loss 卡在 10.397（= log(32768)），模型完全不学习。

修复：在 `init_weights()` 中显式调用 `gamma.data.fill_(1.0)`。

教训：**nanochat 的所有参数初始化必须在 `init_weights()` 中显式完成**，不能依赖 `__init__` 中的默认值。

## 4. 实验设置

| 配置 | 值 |
|------|-----|
| 硬件 | Vast.ai H100 SXM 80GB 单卡 |
| 训练步数 | 1000 steps / 组 |
| 评估间隔 | 每 100 步 |
| Batch size | 524,288 tokens |
| 序列长度 | 2048 |
| 评估指标 | val bpb（验证集 bits-per-byte） |

## 5. 参数量对比

| 模型 | 总参数 | Transformer 矩阵 | Norm Gamma | vs Baseline |
|------|--------|-----------------|------------|-------------|
| Baseline | 286,261,730 | 84,935,088 | 0 | — |
| SwiGLU (S) | 286,261,730 | 84,935,088 | 0 | **精确等参** |
| LearnableNorm (N) | 286,281,698 | 84,935,088 | 19,968 | +0.007% |
| MLA (M) | 281,543,138 | 80,216,496 | 0 | **-1.6% total, -17% attn** |
| S+N+M | 281,563,106 | 80,216,496 | 19,968 | -1.6% total |

## 6. 训练结果

### 6.1 Val BPB 曲线

| Step | Baseline | SwiGLU (S) | LearnableNorm (N) | MLA (M) | S+N+M |
|------|:--------:|:----------:|:-----------------:|:-------:|:-----:|
| 0 | 3.170 | 3.170 | 3.170 | 3.170 | 3.170 |
| 100 | 1.384 | 1.513 | **1.340** | 1.464 | 1.379 |
| 200 | 1.126 | 1.224 | **1.118** | 1.193 | 1.126 |
| 300 | 1.063 | 1.088 | **1.058** | 1.086 | 1.066 |
| 400 | 1.028 | 1.035 | **1.023** | 1.040 | 1.032 |
| 500 | 0.989 | 0.997 | **0.987** | 1.000 | 0.994 |
| 600 | 0.962 | 0.968 | **0.961** | 0.971 | 0.968 |
| 700 | 0.942 | 0.947 | **0.940** | 0.949 | 0.947 |
| 800 | 0.925 | 0.931 | **0.924** | 0.931 | 0.930 |
| 900 | 0.913 | 0.918 | **0.912** | 0.919 | 0.918 |
| **1000** | **0.905** | 0.910 | **0.904** | 0.911 | 0.910 |

### 6.2 最终指标汇总

| 模型 | val bpb @1000 | vs Baseline | 训练时间 | tok/sec | MFU |
|------|:------------:|:-----------:|:--------:|:-------:|:---:|
| **LearnableNorm (N)** | **0.904** | **-0.001** | 16.28m | 530K | 40.7% |
| Baseline | 0.905 | — | 15.85m | 546K | 42.0% |
| SwiGLU (S) | 0.910 | +0.005 | 16.42m | 526K | 40.4% |
| S+N+M | 0.910 | +0.005 | 16.96m | 504K | 37.3% |
| MLA (M) | 0.911 | +0.006 | 15.86m | 546K | 40.3% |

## 7. 分析与结论

### 7.1 各变体分析

**LearnableNorm (N) — 唯一正向改善**
- 在所有 step 上均优于 baseline（step 100 即领先 0.044 bpb）
- 最终 0.904 vs 0.905，改善虽小但一致
- 参数增加仅 0.007%，几乎零成本
- 说明无参数 RMSNorm 确实丢失了少量可学习的特征缩放信息

**SwiGLU (S) — 未达预期**
- 最终 0.910 vs baseline 0.905，略差
- 前期收敛明显更慢（step 100 差 0.13），后期差距缩小至 0.005
- 原因分析：(1) baseline 的 ReLU² 本身有类 gating 效果，与 SwiGLU 功能重叠；(2) 所有超参（LR、init scale 0.4×、weight decay）为 ReLU² 精调，SwiGLU 可能需要不同配置
- 更长训练或超参调优可能改变结果

**MLA (M) — 参数效率良好**
- 最终 0.911 vs baseline 0.905，差 0.006
- 但注意力参数减少 17%，总参数减少 1.6%（省 4.7M 参数）
- 参数效率角度：每参数贡献的 bpb 下降量优于 baseline
- 训练速度与 baseline 持平（矩阵更小但多了投影操作）

**S+N+M 组合 — 无叠加增益**
- 最终 0.910，与 SwiGLU 单独持平，未从 LearnableNorm 获得额外收益
- 说明在当前超参下，组合变体没有协同效应
- SwiGLU 的负面效应抵消了 LearnableNorm 的正向改善

### 7.2 核心结论

1. **LearnableNorm 是最有效的单一改进**：零成本（+0.007% 参数），全程领先 baseline，推荐采纳
2. **SwiGLU 在 ReLU² baseline 下不一定有优势**：文献中的 SwiGLU 优势主要对比 vanilla ReLU/GELU，nanochat 的 ReLU² 已是强 baseline
3. **MLA 是有价值的参数效率方案**：17% 的注意力参数节省换来仅 0.006 的 bpb 代价，适合参数/显存受限场景
4. **架构改进需要配套的超参调优**：SwiGLU 和 MLA 的结果可能受制于为 baseline 精调的超参

### 7.3 推荐架构配置

- **默认推荐**：Baseline + LearnableNorm（`--use-learnable-norm`）
- **参数受限场景**：MLA + LearnableNorm（`--attn-variant mla --use-learnable-norm`）

## 8. 运行方式

```bash
# Baseline
python -m scripts.base_train --depth=12

# SwiGLU
python -m scripts.base_train --depth=12 --mlp-variant swiglu

# LearnableNorm
python -m scripts.base_train --depth=12 --use-learnable-norm

# MLA
python -m scripts.base_train --depth=12 --attn-variant mla

# 全组合
python -m scripts.base_train --depth=12 --mlp-variant swiglu --use-learnable-norm --attn-variant mla

# 运行测试
python -m pytest tests/test_arch_variants.py -v
```

## 9. 训练日志

完整日志保存在 `dev/logs/` 目录下：
- `task3_base1k.log` — Baseline 1000 步
- `task3_S.log` — SwiGLU 1000 步
- `task3_N.log` — LearnableNorm 1000 步
- `task3_M.log` — MLA 1000 步
- `task3_SNM.log` — S+N+M 1000 步
