# Task 7: API 服务化

## 1. 背景与任务目标

**任务**: 将 nanochat 推理封装为生产级 API 服务，实现兼容 OpenAI 格式的 `/v1/chat/completions` 端点，支持流式返回（SSE）。

**解决的问题**: 原有 `/chat/completions` 端点使用自定义格式（`{token: "xxx"}`），无法直接被 OpenAI SDK 或标准工具调用。

## 2. 现有架构理解

### 涉及模块

| 文件 | 职责 |
|------|------|
| `scripts/chat_web.py` | FastAPI 服务器，已有 `/chat/completions`（自定义格式）、WorkerPool 多 GPU 调度、请求校验 |
| `nanochat/engine.py` | 推理引擎，KVCache prefill + decode，tool use（计算器） |

### 原有 API 状态

- `POST /chat/completions`: 自定义 SSE 格式（`data: {"token": "xxx", "gpu": 0}`），仅流式
- `GET /health`: 健康检查
- `GET /stats`: Worker 池状态
- WorkerPool: 异步 worker 队列，支持多 GPU 负载均衡
- 请求校验: 消息数量、长度、temperature/top_k 范围限制

## 3. 设计方案

在现有 FastAPI 应用上新增 OpenAI 兼容端点，保留原有端点不变。

### 新增端点

| 端点 | 说明 |
|------|------|
| `POST /v1/chat/completions` | OpenAI 兼容聊天补全（流式 + 非流式） |
| `GET /v1/models` | 模型列表 |

### 设计决策

1. **复用 WorkerPool 和 generate_stream**: 不重复实现推理逻辑，OpenAI 端点在内部调用已有的 `generate_stream()` 并转换输出格式
2. **复用 validate_chat_request**: OpenAI 请求先转换为内部 `ChatRequest` 再校验，避免重复校验逻辑
3. **model 字段校验**: 请求中的 model 必须匹配 `/v1/models` 返回的 id（"nanochat"），不匹配返回 400
4. **保留原有 `/chat/completions`**: 向后兼容，WebUI 仍使用原有格式

### 备选方案

- 替换原有端点: 会破坏 WebUI 前端，不可取
- 单独 FastAPI 应用: 增加部署复杂度，不必要

## 4. 关键实现说明

### 修改文件

仅修改 `scripts/chat_web.py`。

### 请求格式

```python
class OpenAIChatRequest(BaseModel):
    model: str = "nanochat"          # 必须匹配 /v1/models
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None    # nucleus sampling
    top_k: Optional[int] = None      # 扩展字段（非 OpenAI 标准）
    stream: Optional[bool] = False
```

### 流式响应格式（`stream=True`）

严格遵循 OpenAI SSE 格式：

```
# 首个 chunk: 角色声明
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1776279473,"model":"nanochat","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

# 中间 chunk: 内容增量
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1776279473,"model":"nanochat","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

# 结束 chunk: 停止原因
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1776279473,"model":"nanochat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

# 终止信号
data: [DONE]
```

### 非流式响应格式（`stream=False`）

```json
{
    "id": "chatcmpl-0f8495029b6d",
    "object": "chat.completion",
    "created": 1776279458,
    "model": "nanochat",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "..."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 64,
        "total_tokens": 75
    }
}
```

### Token 计数策略

- **prompt_tokens**: 基于实际 tokenize 结果精确计数（含 bos、special tokens）
- **completion_tokens**: 生成过程中逐 token 累加
- **total_tokens**: prompt + completion

### 容易出错的点

- 流式响应中 `finish_reason` 必须在最后一个 content chunk 之后的独立 chunk 中发送，不能和 content 混在一起
- `data: [DONE]` 不是 JSON，是特殊终止信号
- worker 必须在流式响应结束后（`finally` 块中）释放回 pool，否则 worker 泄漏

## 5. 验证方法

### 远端实测（H100）

全部在远端服务器上实际执行：

```bash
# 1. /v1/models
curl -s http://localhost:8000/v1/models
# => {"object":"list","data":[{"id":"nanochat",...}]}

# 2. 非流式
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nanochat","messages":[{"role":"user","content":"What is 2+2?"}],"stream":false,"max_tokens":64}'
# => 完整 OpenAI 格式响应，含 id/object/created/model/choices/usage

# 3. 流式
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nanochat","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":32}'
# => 首 chunk delta.role -> content chunks -> finish_reason=stop -> [DONE]

# 4. model 校验
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}'
# => HTTP 400: "Model 'gpt-4' not found. Available: nanochat"
```

### Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 非流式
response = client.chat.completions.create(
    model="nanochat",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=64,
)
print(response.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="nanochat",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl 示例

```bash
# 非流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello"}],"stream":true,"max_tokens":64}'
```

## 6. 结果展示

全部端点在远端 H100 上实测通过：

| 测试 | 结果 |
|------|:---:|
| `GET /v1/models` | OK |
| `POST /v1/chat/completions` stream=false | OK（含 id/object/created/model/choices/usage） |
| `POST /v1/chat/completions` stream=true | OK（role -> content -> stop -> [DONE]） |
| model 校验（传 "gpt-4"） | OK（400 错误） |
| top_p 参数传递 | OK |
| 原有 `/chat/completions` 不受影响 | OK |

## 7. 局限性

1. **无认证/rate limiting**: 当前无 API key 校验和请求频率限制（可选加分项未实现）
2. **无 Dockerfile**: 未提供容器化部署（可选加分项未实现）
3. **usage.prompt_tokens 近似**: 基于 tokenizer 编码计数，含 special tokens 但不含 attention mask 开销
4. **仅单模型**: `/v1/models` 固定返回 "nanochat"，不支持多模型切换
5. **不支持 `n` 参数**: OpenAI 的 `n` 参数（多个候选回复）未实现
6. **不支持 `function_calling`/`tools`**: OpenAI 的函数调用接口未实现（nanochat 内部有 calculator tool，但未暴露为 OpenAI 格式）

## 8. 运行说明

```bash
# 启动服务器（需要 GPU）
source .venv/bin/activate && export NANOCHAT_BASE_DIR=/workspace/nanochat_data
python -m scripts.chat_web -i sft -g d12_safety_smallbatch_lradj

# 如在云端，通过 SSH tunnel 访问
ssh -p 19445 -L 8000:localhost:8000 root@<server_ip>

# 测试 API
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

启动后可用端点：
- `http://localhost:8000/` — Web UI（含 temperature 滑块）
- `http://localhost:8000/v1/chat/completions` — OpenAI 兼容 API
- `http://localhost:8000/v1/models` — 模型列表
- `http://localhost:8000/health` — 健康检查
- `http://localhost:8000/chat/completions` — 原有自定义 API（WebUI 使用）

## 9. 修改文件说明

| 文件 | 修改内容 | 为什么改 | 是否影响默认行为 |
|------|---------|---------|:---:|
| `scripts/chat_web.py` | 新增 `OpenAIChatRequest` model、`/v1/chat/completions` 端点（流式+非流式）、`/v1/models` 端点、model 校验、top_p 透传 | 题目要求 OpenAI 兼容 API | 否（原有端点不变） |
