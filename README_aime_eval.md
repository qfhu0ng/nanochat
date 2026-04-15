# AIME 2024/2025 评估任务 — 交付文档

## 1. 任务背景

AIME（American Invitational Mathematics Examination）是美国顶级数学竞赛，每年 30 道题，答案均为 0-999 的整数。AIME 被广泛用于评估 LLM 的数学推理能力。

本任务为 nanochat 添加 AIME 2024 和 AIME 2025 评估，符合现有 Task 接口规范，可通过 `scripts/chat_eval.py` 直接运行。

## 2. 设计思路

### 2.1 为什么选择 generative 评估

AIME 是开放式数学题，不是选择题。模型需要生成完整的解题过程并给出最终数字答案，因此使用 `eval_type = 'generative'`，与 GSM8K 的评估方式一致。

### 2.2 答案提取策略（四层优先级）

| 优先级 | 格式 | 示例 | 说明 |
|--------|------|------|------|
| 1 | `\boxed{n}` | `\boxed{204}` | LaTeX 格式，数学竞赛标准输出 |
| 2 | `#### n` | `#### 204` | GSM8K 格式，nanochat SFT 训练中已学会 |
| 3 | `final answer is n` | `the final answer is 204` | 自然语言表述 |
| 4 | 最后独立整数 | `...so we get 204` | 回退兜底，取最后一个 0-999 整数 |

所有提取结果都经过范围校验：`0 <= answer <= 999`，超范围判定为提取失败。

### 2.3 为什么不加入 ChatCORE 主指标

AIME 极其困难（GPT-4 ~30%，小模型接近 0%），加入 `all_tasks` 会拉低 ChatCORE 分数。因此仅注册在 `task_module` 字典中，可独立运行评估，不影响现有指标体系。

### 2.4 Prompt 设计

在问题后附加引导文本：
```
Note: This is an AIME problem. The answer is an integer between 0 and 999.
Show your work, then give your final answer after ####.
```
原因：引导模型使用 `####` 格式输出（与 GSM8K SFT 训练格式一致），提高答案提取成功率。

## 3. 评估指标

- **主指标**：Accuracy（exact match），预测整数 == 真实整数即为正确
- **`-n > 1` 时**：pass@k 语义 — 对每道题采样 k 次，**任意一次**正确即算通过
- **baseline**：0.0（答案空间 0-999，随机猜对概率 ~0.1%，可忽略）

## 4. 数据集

| 数据集 | 题数 | HuggingFace ID | 字段 |
|--------|------|----------------|------|
| AIME 2024 | 30 | `HuggingFaceH4/aime_2024` | problem, answer(str), solution |
| AIME 2025 | 30 | `MathArena/aime_2025` | problem, answer(int), problem_type |

### 数据可复现性

- 本地缓存：`dev/aime_problems.jsonl`（60 题完整备份）
- 若 HuggingFace 数据集下线，可从本地缓存恢复

## 5. 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `tasks/aime.py` | 新增 | AIME Task 类（数据加载 + 答案提取 + evaluate） |
| `scripts/chat_eval.py` | 修改 | 注册 `AIME-2024` 和 `AIME-2025`，添加 import |
| `tests/test_aime_task.py` | 新增 | 37 个单元测试（提取 + 范围校验 + evaluate + 数据加载） |
| `dev/aime_problems.jsonl` | 新增 | 60 题本地缓存 |
| `README_aime_eval.md` | 新增 | 本文档 |

## 6. 运行方式

```bash
# 评估 AIME 2024
python -m scripts.chat_eval -i sft -a AIME-2024

# 评估 AIME 2025
python -m scripts.chat_eval -i sft -a AIME-2025

# 两个年份一起评估
python -m scripts.chat_eval -i sft -a "AIME-2024|AIME-2025"

# 多次采样（pass@8），temperature=0.7
python -m scripts.chat_eval -i sft -a AIME-2024 -n 8 -t 0.7

# 运行单元测试
python -m pytest tests/test_aime_task.py -v
```

## 7. 验证结果

### 单元测试
```
37 passed in 21.14s
```

覆盖：
- 答案提取 4 层优先级（19 个 case）
- 范围校验（4 个 case）
- evaluate 逻辑（5 个 case）
- 数据加载完整性（9 个 case）

### 端到端评估结果

在 Vast.ai H100 SXM 80GB 上，使用 SFT 后的 d12 模型（~286M 参数）评估：

| 评估集 | 正确数 | 总题数 | 准确率 |
|--------|--------|--------|--------|
| **AIME-2024** | 0 | 30 | **0.00%** |
| **AIME-2025** | 1 | 30 | **3.33%** |

这符合预期 — 即使 GPT-4 在 AIME 上也仅约 30%，0.3B 小模型基本无法进行有效的数学推理。

### 模型输出样例

**AIME-2024 第 1 题**（正确答案：204）

> **输入**: Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop...
>
> **模型输出**: To find the number of minutes the walk takes Aya, we need to calculate the total time spent in the coffee shop and then divide it by the total time spent in the coffee shop... *(逻辑混乱，反复重复，无法有效推导)*

### AIME 评估的价值

尽管小模型准确率接近 0%，AIME 评估仍有价值：
1. 提供高难度数学推理 benchmark，与其他模型可直接对比
2. 随着模型规模增大，可观察到准确率提升趋势
3. AIME-2025 意外答对 1 题（3.33%），说明评估 pipeline 端到端工作正常

### 评估日志

完整日志保存在 `dev/logs/` 目录下：
- `aime_2024_eval.log`
- `aime_2025_eval.log`

## 8. 已知限制与改进方向

### 已知限制
- 数据集仅有 30 题/年，样本量小，评估结果方差较大
- 小模型基本无法解答 AIME 题目，评估意义有限
- 答案提取的回退策略（最后独立整数）可能误抓推导中间数

### 改进方向
- 支持更多年份的 AIME 数据（2020-2023）
- 添加分类统计（代数/几何/数论/组合）
- 实现 chain-of-thought prompting 提升提取准确率
- 支持从本地 JSONL 缓存加载（当 HF 不可用时）
