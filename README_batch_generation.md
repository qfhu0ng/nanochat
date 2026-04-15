# Task 4: Batch Generation

## 任务背景

原始 `Engine.generate()` 仅支持单个 prompt 的多采样生成（同一 prompt → N 个独立序列）。Task 4 实现**不同长度 prompt 的 batched decode 推理**，正确处理 ragged-length 序列的 KV cache 和位置编码。

**术语说明：** 本方案实现的是"ragged-length prompts 的 batched decode"——将多个不同长度的 prompt 合并为一个 batch 并行生成。这不同于 vLLM 式的 queue-level dynamic batching（请求级动态合批），后者需要调度队列和请求中途加入/退出机制。

## 核心挑战

不同 prompt 长度 → prefill 后各 batch element 的 KV cache 位置不同 → RoPE 位置编码需要支持 per-element offset。

## 设计方案：Sequential Prefill + Batched Decode + Per-element RoPE

1. **Sequential Prefill**: 逐个 prefill 每个 prompt（batch=1），RoPE 和 attention 天然正确
2. **Copy to Batched Cache**: 每个 prefill 结果复制到统一的 batched KV cache 对应 slot
3. **Batched Decode**: 并行 decode，使用 per-element position 处理 RoPE 和 attention mask

### 为什么不用 Padding？

- **Left-padding**: causal attention 让真实 token attend 到 PAD token，破坏 KV 值
- **Right-padding**: 需要复杂的 per-element attention mask，且 FA3 不支持

## 实现细节

### 1. KVCache 扩展 (`nanochat/engine.py`)

新增两个方法：
- `is_uniform()`: 检查所有 batch element 是否在同一 cache 位置
- `copy_from_single(other, batch_idx)`: 将 batch=1 的 cache 复制到指定 batch slot

### 2. Per-element RoPE (`nanochat/gpt.py`)

`forward()` 中 RoPE 切片逻辑扩展为三分支：
- **无 cache**: `T0=0`，标准切片
- **Uniform cache**: `T0=cache.get_pos()`，标准切片（原有逻辑）
- **Non-uniform cache**: 仅 T=1 decode step，用 `cache_seqlens` 做 per-element indexing

### 3. SDPA Fallback (`nanochat/flash_attention.py`)

SDPA 路径扩展 non-uniform 分支（T_new=1 only）：
- Per-element cache insertion（循环插入，非 batch slice）
- Length mask: 每个 element 只 attend 到 `[0, cache_seqlens[b]+1)` 范围
- 可选 sliding window mask

FA3 路径（Hopper GPU）原生支持 per-element `cache_seqlens`，无需修改。

### 4. Engine 重构 (`nanochat/engine.py`)

- 抽取 `_decode_loop()`: 共享的 decode 循环（sampling + tool-use 状态机 + completion tracking）
- `generate()`: 重构为调用 `_decode_loop()`，行为 100% 等价
- `generate_multi(prompts, ...)`: 新增，sequential prefill + batched decode
- `generate_multi_batch(prompts, ...)`: 新增，非 streaming 便捷方法

## 修改文件清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `nanochat/engine.py` | KVCache +2 方法, Engine +3 方法 | 核心逻辑 |
| `nanochat/gpt.py` | forward() RoPE 切片 ~10 行 | per-element 位置编码 |
| `nanochat/flash_attention.py` | SDPA fallback ~30 行 | non-uniform cache 支持 |
| `scripts/bench_batch.py` | 新文件 | 性能测试脚本 |
| `tests/test_engine.py` | +6 个测试 | batch generation 正确性测试 |
| `README_batch_generation.md` | 新文件 | 本文档 |

## 运行方式

### 正确性 + 性能测试

```bash
# Greedy decode (temperature=0)，自动验证 batch 输出 == sequential 输出
python -m scripts.bench_batch -i sft -g <model_tag> --max-tokens 64 --temperature 0.0

# 自定义 batch sizes
python -m scripts.bench_batch -i sft -g <model_tag> --batch-sizes 1,2,4,8,16

# 带采样的性能测试
python -m scripts.bench_batch -i sft -g <model_tag> --temperature 0.6 --max-tokens 128
```

### 现有测试回归

```bash
# Engine.generate() 与 model.generate() 一致性测试（原有）
python -m nanochat.engine
```

## Benchmark 结果

测试环境：H100 SXM，d12 SFT 模型（d12_safety_smallbatch_lradj），greedy decode (temperature=0)，max_tokens=64。

8 个不同长度的 prompt（11-20 tokens），涵盖短问题到长指令。

| Batch Size | Sequential tok/s | Batched tok/s | 加速比 | 正确性 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 154 | 154 | 1.00x | PASS |
| 2 | 155 | 294 | 1.89x | PASS |
| 4 | 155 | 574 | 3.71x | PASS |
| 8 | 155 | 1055 | **6.80x** | PASS |

**分析：**
- 吞吐量近似线性增长，batch=8 时达到 6.8x 加速
- d12 模型在 H100 上 decode 阶段是 memory-bandwidth bound，单 prompt 远未打满计算资源，batching 几乎免费利用了空闲的计算带宽
- Sequential per-prompt latency 稳定在 ~413ms（p50=p90），batched 总延迟仅 466ms（8 个 prompt 并行完成）

### 回归测试

`python -m nanochat.engine` 输出 `Match: True`，确认重构后的 `Engine.generate()` 与 `model.generate()` 完全一致。

## 开发过程中遇到的问题

### 1. Completed row 继续产生 junk tokens

**现象：** batch=8 greedy 正确性校验 FAIL，prompt 5 在 sequential 模式下生成 44 个 token（遇到 `assistant_end` 停止），但 batch 模式下报告生成了 64 个 token。

**根因：** 不是生成逻辑的 bug。在 `_decode_loop` 中，某个 row 遇到 `assistant_end` 后标记 `completed=True`，但其他 row 还在继续。completed row 的 token 仍然被 feed 进模型（占用 batch slot），产生无意义的后续 token。benchmark 的收集函数 `run_batched` 没有过滤已完成 row，把这些 junk tokens 也计入了结果。

而 sequential 模式下每个 prompt 独立运行（batch=1），遇到 `assistant_end` 后 `all(completed)` 立即为 true，循环退出，不会产生多余 token。

**修复：** 在 benchmark 的 `run_sequential_timed` 和 `run_batched_timed` 中都加入 completed row 过滤，遇到 `assistant_end`/`bos` 后停止收集该 row 的 token。这与 `generate_batch()` / `generate_multi_batch()` 的处理方式一致。

**启示：** streaming API（`generate_multi`）的调用方需要自行处理 per-row completion；非 streaming API（`generate_multi_batch`）已内置处理。

### 2. KV cache 显存分配策略不一致

**现象：** `generate()` 在 `max_tokens=None` 时用 `config.sequence_len` 作为 KV cache 长度，但 `generate_multi()` 最初实现用 `max_prompt_len + config.sequence_len`，导致显存占用翻倍。

**修复：** 统一为 `max_tokens is None` 时使用 `config.sequence_len`，与 `generate()` 一致。

## 验证方案

1. **Greedy 一致性**: `temperature=0` 时，batch 输出与逐个 sequential 生成 token-by-token 完全一致（所有 batch size 均 PASS）
2. **RoPE 正确性**: 通过 greedy 一致性间接验证——如果 RoPE 位置错误，生成结果必然不同
3. **吞吐量**: 不同 batch size (1/2/4/8) 下的 tok/s 对比，含 per-prompt p50/p90 latency
4. **单元测试**: 14 个测试全部通过，覆盖 KVCache（is_uniform, copy_from_single, prefill）、generate_multi（greedy 一致性、max_tokens、单 prompt 等价性）等
5. **回归**: `python -m nanochat.engine` 原有测试通过（`Match: True`，generate() 行为不变）

## 设计约束与限制

- Non-uniform RoPE/SDPA 路径**仅支持 T=1 decode step**（assert 保护），不支持 non-uniform prefill
- 已完成的 row 仍占用 batch slot（不做动态压缩），适用于 batch size ≤ 16 的场景
- SDPA non-uniform 路径中 cache insertion 使用 Python 循环（B 次），batch size 较大时可能有开销
- 不是服务端 queue-level dynamic batching，是静态 batch 的 ragged-length decode

## 后续可改进方向

- 动态 batch 压缩：已完成的 row 提前退出，释放 compute
- Continuous batching：请求级动态合批，新请求中途加入
- SDPA non-uniform 路径的 vectorized cache insertion（避免 Python 循环）
- Prefill 也做 batching（right-padding + custom mask），减少 kernel launch 开销
