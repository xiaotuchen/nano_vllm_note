# nano-vllm 代码结构与调用逻辑

## 目录树结构

```
nano-vllm/
├── README.md                # 项目介绍与用法
├── LICENSE                  # 许可证
├── nanovllm/
│   ├── __init__.py          # 包导入入口
│   ├── llm.py               # LLM主类，推理API入口
│   ├── sampling_params.py   # 采样参数定义
│   ├── config.py            # 配置参数定义
│   ├── engine/
│   │   ├── llm_engine.py    # 推理主流程，API接口
│   │   ├── model_runner.py  # 模型加载与分布式推理
│   │   ├── scheduler.py     # 推理调度与分块分配
│   │   ├── sequence.py      # 输入序列对象
│   │   ├── block_manager.py # KV缓存分块管理
│   ├── layers/
│   │   ├── linear.py        # 线性层（并行/分片等）
│   │   ├── attention.py     # 注意力层与KV缓存写入
│   ├── models/
│   │   ├── qwen3.py         # Qwen3模型结构定义
│   ├── utils/
│   │   ├── context.py       # 推理上下文管理
│   │   ├── loader.py        # 权重加载工具
│   └── ...                  # 其他辅助模块
```

---

## 主要文件作用说明

### nanovllm/__init__.py
- 包导入入口，暴露 `LLM` 和 `SamplingParams`。

### nanovllm/llm.py
- LLM主类，继承自 `engine/llm_engine.py`，对外提供推理API。

### nanovllm/sampling_params.py
- 定义采样参数（如温度、最大生成长度等）。

### nanovllm/config.py
- 定义模型、推理相关的配置参数。

### nanovllm/engine/llm_engine.py
- 推理主流程，负责模型初始化、请求管理、调度、采样和输出。
- 主要API：`add_request`、`step`、`generate`。

### nanovllm/engine/model_runner.py
- 模型加载与分布式推理，负责权重加载、KV缓存分配、实际推理。
- 支持多进程/多卡并行。

### nanovllm/engine/scheduler.py
- 推理调度器，管理输入序列状态，分配分块资源。

### nanovllm/engine/sequence.py
- 输入序列对象，管理token、分块、状态等。

### nanovllm/engine/block_manager.py
- KV缓存分块管理，支持块复用和缓存命中。

### nanovllm/layers/linear.py
- 线性层实现，支持分布式并行（行/列/合并/QKV等）。

### nanovllm/layers/attention.py
- 注意力层实现，包含KV缓存写入、FlashAttention加速等。

### nanovllm/models/qwen3.py
- Qwen3模型结构定义。

### nanovllm/utils/context.py
- 推理上下文管理。

### nanovllm/utils/loader.py
- 权重加载工具。

---

## 主要调用逻辑

1. **初始化**：
   - 用户通过 `LLM(model_path, ...)` 创建模型对象。
   - 初始化分布式进程、模型、分块调度器、分词器等。

2. **请求添加**：
   - 通过 `add_request(prompt, sampling_params)` 添加推理请求。
   - 构造 `Sequence` 对象，加入调度队列。

3. **调度与分块**：
   - `Scheduler` 根据资源和序列状态分配分块，决定哪些序列进入推理。
   - `BlockManager` 管理KV缓存分块。

4. **模型推理**：
   - `ModelRunner` 执行模型前向，采样输出token，管理KV缓存。
   - 多进程/多卡通过共享内存和分布式通信同步推理结果。

5. **采样与输出**：
   - 采样器根据logits采样token，`Scheduler`更新序列状态，回收资源。
   - `LLMEngine.generate` 汇总输出，返回文本和token序列。

---

如需某个模块/类/函数详细解读，可进一步指定。
