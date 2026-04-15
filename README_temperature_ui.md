# Task 6: Temperature 采样控制 UI

## 1. 背景与任务目标

**任务**: 为 nanochat Web UI 添加采样参数控制，包括 temperature 滑块和可选的 top-p、top-k 控制。

**解决的问题**: 原有 UI 仅支持通过 `/temperature` 和 `/topk` 斜杠命令调整采样参数，用户无法直观地看到和调整当前设置。

## 2. 现有架构理解

### 涉及模块

| 文件 | 职责 |
|------|------|
| `nanochat/ui.html` | 聊天 UI 前端（纯 HTML/CSS/JS，无框架） |
| `scripts/chat_web.py` | FastAPI 后端，已支持 temperature/top_k 参数透传 |
| `nanochat/engine.py` | 推理引擎，`sample_next_token()` 实现采样逻辑 |

### 原有状态

- **后端**: `chat_web.py` 已接收 temperature (0.0-2.0) 和 top_k (0-200) 参数
- **前端**: 通过 JS 变量 `currentTemperature` 和 `currentTopK` 在请求中传递，但仅能通过 `/temperature <value>` 和 `/topk <value>` 命令修改
- **Top-p**: 整个链路均不支持（engine.py 中无 nucleus sampling 实现）

## 3. 设计方案

### UI 设计

在 header 右侧添加齿轮按钮，点击展开/收起设置面板。面板包含三个滑块控件：

- **Temperature** (0.0-1.0, 步长 0.1): 控制输出随机性
- **Top-p** (0.0-1.0, 步长 0.05): Nucleus sampling，1.0 时显示 "off" 表示禁用
- **Top-k** (0-200, 步长 1): Top-k 过滤，0 时显示 "off" 表示禁用

### 设计决策

1. **齿轮按钮 + 折叠面板**: 不占用聊天区域空间，需要时展开
2. **与斜杠命令双向同步**: 滑块改值更新 JS 变量，斜杠命令改值也更新滑块位置
3. **top-p = 1.0 视为禁用**: 符合 OpenAI API 语义（top_p=1.0 等于不过滤）

### 备选方案

- 侧边栏设置面板: 实现更复杂，对小屏不友好
- 仅扩展斜杠命令: 不满足题目"滑块"要求

## 4. 关键实现说明

### 新增/修改文件

| 文件 | 修改内容 |
|------|---------|
| `nanochat/engine.py` | `sample_next_token()` 新增 `top_p` 参数，实现 nucleus sampling |
| `nanochat/ui.html` | header 齿轮按钮 + 设置面板（3 个滑块）+ JS 函数 |
| `scripts/chat_web.py` | `ChatRequest`/`OpenAIChatRequest` 新增 `top_p` 字段，透传到 `generate_stream()` |

### Top-p (Nucleus Sampling) 实现

```python
# engine.py: sample_next_token()
if top_p is not None and 0.0 < top_p < 1.0:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # 保留累积概率首次超过 top_p 的 token
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[remove_mask] = float('-inf')
    logits.scatter_(1, sorted_idx, sorted_logits)
```

Top-k 和 top-p 可以组合使用（先 top-k 过滤，再 top-p 过滤），符合 HuggingFace/OpenAI 的标准行为。

### 容易出错的点

- top-p 的 cumulative mask 需要保留首个超过阈值的 token（用 `cumulative - current >= top_p`），否则可能把所有 token 都过滤掉
- top_p=1.0 不应做任何过滤（前端发 `null`，后端跳过）
- 滑块和斜杠命令需要双向同步，否则用户看到的值和实际使用的值不一致

## 5. 验证方法

### 功能验证

1. **UI 存在性**: `curl` 获取 `/` 页面，确认包含 `settingsPanel`、`tempSlider`、`toppSlider`、`topkSlider` 元素
2. **参数传递**: 通过 API 发送带 temperature/top_p/top_k 的请求，确认不报错且响应正常
3. **温度效果**: temperature=0.0 应产生确定性输出（相同 seed 相同结果）
4. **浏览器验证**: SSH tunnel 打开 UI，手动拖动滑块确认交互流畅

### 已验证

- 远端 H100 启动 WebUI，通过 SSH tunnel 在浏览器中确认：
  - 齿轮按钮正常展开/收起
  - 三个滑块可拖动，数值实时更新
  - 聊天功能正常，参数生效

## 6. 结果展示

- Temperature 滑块: 0.0-1.0，步长 0.1，默认 0.8
- Top-p 滑块: 0.0-1.0，步长 0.05，默认 off (1.0)
- Top-k 滑块: 0-200，步长 1，默认 50，0 时显示 "off"
- 齿轮按钮在 header 右侧，点击展开设置面板
- 与 `/temperature`、`/topk` 斜杠命令双向同步

## 7. 局限性

1. **无 top-p 斜杠命令**: 仅通过滑块控制，未添加 `/topp` 命令（可后续补充）
2. **temperature 前端限制 0-1，后端支持 0-2**: 前端按题目要求限制 0-1，后端保留更宽范围（API 用户可用 0-2）
3. **无实时预览**: 调整参数后需发送新消息才能看到效果，无法实时预览采样分布变化
4. **移动端适配**: 设置面板在极窄屏幕上可能需要滚动

## 8. 运行说明

```bash
# 启动 WebUI（需要 GPU）
source .venv/bin/activate && export NANOCHAT_BASE_DIR=/workspace/nanochat_data
python -m scripts.chat_web -i sft -g d12_safety_smallbatch_lradj

# 如在云端，通过 SSH tunnel 访问
ssh -p 19445 -L 8000:localhost:8000 root@<server_ip>
# 浏览器打开 http://localhost:8000
```

操作步骤：
1. 点击右上角齿轮图标展开设置面板
2. 拖动 Temperature/Top-p/Top-k 滑块
3. 在输入框输入消息并发送，观察采样效果变化

## 9. 修改文件说明

| 文件 | 修改内容 | 为什么改 | 是否影响默认行为 |
|------|---------|---------|:---:|
| `nanochat/engine.py` | `sample_next_token()` 新增 `top_p` 参数，实现 nucleus sampling；`generate()` 透传 `top_p` | 后端无 top-p 支持 | 否（`top_p=None` 时行为不变） |
| `nanochat/ui.html` | header 齿轮按钮 + 折叠面板（3 滑块）+ JS 函数 + 斜杠命令同步 | 题目要求可视化控件 | 否（默认值与原来一致） |
| `scripts/chat_web.py` | `ChatRequest`/`OpenAIChatRequest` 新增 `top_p` 字段，`generate_stream()` 透传，新增 top_p 校验 | 透传 top_p 到 engine | 否（`top_p=None` 时不传） |
