# Task 5: Speculative Decoding

## 算法

Speculative decoding（Leviathan et al., 2023; Chen et al., 2023）通过小型 **draft 模型** 快速猜测 K 个候选 token，再由 **target 模型** 一次 forward pass 验证，加速自回归推理。

**核心保证：** 无论 draft 模型质量如何，输出分布与 target 模型的标准 AR decode **数学上完全一致**。

### 每轮流程

1. **Draft 阶段：** Draft 模型自回归生成 K 个 token（K × T=1 forward）
2. **Verify 阶段：** Target 模型一次 forward pass 处理 K 个 draft token（1 × T=K）
3. **接受/拒绝：**
   - **Greedy (temp=0)：** `argmax(target_logits) == draft_token` 则接受，否则用 target 的 argmax 纠正
   - **Sampling (temp>0)：** 以 `min(1, p(x)/q(x))` 概率接受，拒绝时从 `norm(max(0, p-q))` 重采样
4. **产出：** 每轮 1 ~ K+1 个 token（接受的 draft tokens + 1 个纠正/bonus token）

### Verify 索引映射

| 校验 draft_tokens[i] | Logits 来源 |
|---|---|
| i=0 | `target_logits`（上一轮保存的） |
| i=1..K-1 | `verify_logits[:, i-1, :]` |
| **bonus** | `verify_logits[:, K-1, :]` |

### 每轮开销

K × draft T=1 + 1 × target T=K + 1 × target T=1 + 1 × draft T=1

## 实现

### 单一过滤函数 (`_apply_sampling_filter`)

**分布一致性的关键：** Draft 采样（q）和 target 验证（p）使用**同一个函数**：

```python
_apply_sampling_filter(logits, temperature, top_k, top_p) → probs
```

过滤链：`temp scaling → top_k → top_p (nucleus) → softmax`

`sample_next_token` 重构为内部调用此函数，确保单一代码路径。

### KV Cache 回滚

接受/拒绝后，两个 cache 回滚：

1. `restore_state()` → 恢复到 draft 前的位置（seqlens + prev_embedding）
2. `cache_seqlens += n_accepted`（verify/draft forward 写入的 KV 条目仍然有效）
3. 通过 `_compute_prev_embedding(model, last_accepted_token)` 重算 `prev_embedding`
4. 将 correction/bonus token 通过两个模型 forward（T=1）

### Smear 状态恢复

nanochat 使用 "smear" 机制将前一个 token 的归一化 embedding 混入当前 token。回滚后需要重算 `prev_embedding`：

```python
prev_embedding = embed_norm(wte(token_id))  # 若 embed_norm 为 None 则用 rms_norm
```

此外，gpt.py 的 T>1 + KV cache 代码路径原本不对 position 0 做 smear（因为 prefill 时 position 0 没有前序 token）。Speculative verify 在 decode 中途做 T=K forward，position 0 需要用缓存的 `prev_embedding` 做 smear，因此新增 3 行修复。

### 采样参数

支持 `generate()` 的全部采样参数：
- `temperature`：0.0（greedy）或 > 0（sampling）
- `top_k`：top-k 过滤
- `top_p`：nucleus sampling

## API

```python
engine = Engine(target_model, tokenizer)

# 流式生成
for token_column, token_masks in engine.generate_speculative(
    tokens, draft_model, K=4, max_tokens=256,
    temperature=0.7, top_k=50, top_p=0.9, seed=42
):
    tok = token_column[0]    # 每次 yield 一个 token
    is_draft = token_masks[0]  # 1=draft 接受, 0=纠正/bonus

# 统计信息（生成结束后可读）
stats = engine.speculative_stats
# {'total_draft': N, 'total_accepted': M, 'total_rounds': R}
acceptance_rate = stats['total_accepted'] / stats['total_draft']
```

## Benchmark

### 实验配置

- **Target：** d24 SFT（1.38B params, 24 层, n_embd=1536）
- **Draft：** d12 SFT（286M params, 12 层, n_embd=768）或 d8 SFT（~90M params, 8 层, n_embd=512）
- **硬件：** 1×H100 80GB SXM（HBM3, ~3.35 TB/s 带宽）
- **脚本：** `python -m scripts.bench_speculative --target-tag <tag> --draft-tag <tag>`
- **标准 AR baseline：** d24 上 ~77 tok/s

### 1. 正确性 — 分布一致性验证

**Greedy (temp=0) 强一致性：**
- max_tokens=64：所有 K 值、两个 draft 模型（d12, d8）均 **PASS**
- max_tokens=256：单个 prompt 在 token 218 处分歧（所有 K 值一致）——原因为标准 `generate()` 有 calculator 状态机会注入 forced tokens，`generate_speculative()` 按设计不处理 tool-use；以及 FA3 kernel 在 T=K 与 T=1 之间的数值路径差异

**Sampling (temp>0) 统计一致性：**
- 小词表 MockModel 单元测试：2000 样本 KL 散度 **0.0016**（远在统计置信区间内）
- p 和 q 由同一个 `_apply_sampling_filter` 函数计算——单一代码路径保证理论分布一致

### 2. 加速实测表 — d12 (draft) + d24 (target)

| K | temp | max_tok | wall_ms | tok/s | speedup | accept% | avg_acc/round |
|---|---|---|---|---|---|---|---|
| 2 | 0.0 | 64 | 9630 | 53.2 | 0.70x | 74.5% | 1.5/2 |
| 4 | 0.0 | 64 | 9428 | 54.3 | 0.72x | 56.8% | 2.2/4 |
| 6 | 0.0 | 64 | 9223 | 55.5 | 0.73x | 51.1% | 3.0/6 |
| 8 | 0.0 | 64 | 9969 | 51.4 | 0.68x | 43.6% | 3.3/8 |
| 2 | 0.7 | 64 | 10735 | 47.7 | 0.60x | 62.1% | 1.2/2 |
| 4 | 0.7 | 64 | 9506 | 53.9 | 0.68x | 56.1% | 2.2/4 |
| 6 | 0.7 | 64 | 10909 | 46.9 | 0.59x | 41.6% | 2.4/6 |
| 8 | 0.7 | 64 | 11169 | 45.8 | 0.58x | 38.2% | 3.0/8 |
| 4 | 0.0 | 256 | 28324 | 56.1 | 0.76x | 56.9% | 2.1/4 |
| 4 | 0.7 | 256 | 29658 | 51.9 | 0.67x | 56.8% | 1.7/4 |

### 加速实测表 — d8 (draft) + d24 (target)

| K | temp | max_tok | wall_ms | tok/s | speedup | accept% | avg_acc/round |
|---|---|---|---|---|---|---|---|
| 2 | 0.0 | 64 | 9002 | 56.9 | 0.75x | 64.7% | 1.3/2 |
| 4 | 0.0 | 64 | 8828 | 58.0 | 0.77x | 45.8% | 1.8/4 |
| 6 | 0.0 | 64 | 8989 | 57.0 | 0.76x | 38.3% | 2.2/6 |
| 8 | 0.0 | 64 | 9813 | 52.2 | 0.69x | 30.9% | 2.4/8 |
| 2 | 0.7 | 64 | 9520 | 53.8 | 0.66x | 58.1% | 1.1/2 |
| 4 | 0.7 | 64 | 9599 | 51.5 | 0.65x | 38.9% | 1.4/4 |
| 4 | 0.0 | 256 | 26286 | 60.5 | 0.82x | 45.9% | 1.7/4 |
| 4 | 0.7 | 256 | 31140 | 51.7 | 0.64x | 42.3% | 1.6/4 |

### 3. 场景分析 — 加速条件与实测对照

#### 加速的理论条件

Speculative decoding 每轮产出 `α·K + 1` 个 token（α = acceptance rate），耗时 `K·t_d + t_T(K) + t_T(1) + t_d + overhead`。加速条件为：

```
(α·K + 1) / (K·t_d + t_T(K) + t_T(1) + t_d + overhead) > 1 / t_T(1)
```

其中 `t_d` = draft T=1 耗时，`t_T(1)` = target T=1 耗时，`t_T(K)` = target T=K 耗时。

Decode 阶段是 memory-bandwidth bound，`t_T(K) ≈ t_T(1)`（读同样多的 KV cache）。简化后加速条件为：

```
α > (K·c + 1 + overhead/t_T) / K    其中 c = t_d / t_T(1)
```

即：**acceptance rate 必须足够高，且 draft/target 耗时比 c 必须足够小**。

#### 实测数据代入

从 benchmark 估算单次 forward 耗时（d24 标准 AR: ~77 tok/s → `t_T ≈ 13ms`）：

| 配置 | t_d (ms) | t_T (ms) | c = t_d/t_T | K | α | 加速所需最低 α |
|---|---|---|---|---|---|---|
| d12+d24 | ~5 | ~13 | 0.38 | 4 | 56.8% | ~63% |
| d8+d24 | ~3 | ~13 | 0.23 | 4 | 45.8% | ~48% |

d12+d24 的 acceptance rate（56.8%）低于阈值（63%），因此无加速。d8+d24 的 α（45.8%）也低于阈值（48%），刚好卡在临界以下。

#### 何时加速明显

**1. Target 模型更大（>10B params）时：**

大模型 T=1 decode 慢（如 7B 在 A100 上 ~30ms），而小 draft（68M）只需 ~1ms，c = 0.03。此时加速阈值降至 ~28%，绝大多数 draft 都能满足。典型加速 2-3x。

**2. 低带宽 GPU（消费级）时：**

| GPU | 带宽 | d24 T=1 估计 | c (d8 draft) | 阈值 α |
|---|---|---|---|---|
| H100 SXM | 3.35 TB/s | ~13ms | 0.23 | ~48% |
| A100 | 2.0 TB/s | ~22ms | 0.14 | ~39% |
| RTX 4090 | 1.0 TB/s | ~44ms | 0.07 | ~32% |
| RTX 3090 | 0.94 TB/s | ~47ms | 0.06 | ~31% |

低带宽 GPU 上 target decode 更慢，draft 开销占比更小，加速阈值更低。RTX 4090 上 α > 32% 即可加速——d8+d24 的 45.8% 足够，预期 ~1.4x 加速。

**3. Draft 与 target 分布匹配度更高时：**

同一模型不同量化版本（如 FP16 target + INT4 draft）共享相同的训练数据和架构，acceptance rate 通常 >80%。此时即使在 H100 上也能获得 1.5-2x 加速。

**4. 本实验为何不加速：**

- H100 带宽过高 → target T=1 已经很快（13ms），draft 开销占比大
- d8/d12 与 d24 架构相同但训练数据/步数不同（d8 仅 1280 步 pretrain）→ 分布差异大 → α 低
- 模型未 compile → 额外的 Python/框架开销放大了每轮固定成本

## d24 模型评估结果

训练配置参照 `speedrun.sh`：`--depth=24 --target-param-data-ratio=8 --fp8`，8×H100 SXM，总训练时间 99 分钟。

### Pretrain

| 指标 | d12 | d24 |
|---|---|---|
| 参数量 | 286M | 1.38B |
| Val bpb | 0.868 | **0.716** |
| CORE | — | **0.267** |
| 训练步数 | 5000 | 5568 |
| 训练时间 | 90m (1×H100) | 99m (8×H100) |

### SFT + Safety

| 指标 | d12 SFT | d24 SFT |
|---|---|---|
| Val bpb | 0.356 | **0.274** |
| ChatCORE | 0.254 | **0.365** |
| Safety 拒绝率 | 4% (baseline) | **12%** |

### AIME

| 数据集 | d12 | d24 |
|---|---|---|
| AIME-2024 | 0/30 (0%) | 0/30 (0%) |
| AIME-2025 | 1/30 (3.3%) | 1/30 (3.3%) |

AIME 结果与 d12 一致——1.38B 规模的模型无法解决竞赛数学题。

## 已知限制

1. **Batch size = 1：** 不与 `generate_multi()` 批量生成组合
2. **不处理 tool-use：** Calculator/python tool token 当作普通 token（无 `<|python_start|>` 状态机）
3. **同 GPU：** 两个模型必须在同一 GPU 上（无模型并行）
4. **同词表：** Draft 和 target 必须共享 tokenizer（nanochat 天然满足）
5. **Draft 质量影响速度不影响正确性：** Draft 质量差 → acceptance rate 低 → 加速少，但输出始终是 target 分布

## 修改文件

| 文件 | 改动 |
|---|---|
| `nanochat/engine.py` | `_apply_sampling_filter`, `_rms_norm_fallback`, `_get_kv_model_kwargs_from`, `_compute_prev_embedding`, `KVCache.save_state/restore_state`, `Engine.generate_speculative` |
| `nanochat/gpt.py` | Smear 修复：T>1 KV cache 路径对 position 0 应用 `prev_embedding`（3 行） |
| `scripts/bench_speculative.py` | 新文件：benchmark 脚本 |
| `tests/test_engine.py` | +6 个 speculative decoding 测试 |
| `README_speculative_decoding.md` | 本文档 |

## 参考文献

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML.
- Chen, C., et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.
