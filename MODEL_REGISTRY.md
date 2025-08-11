# 模型注册系统使用说明

## 概述

nano-vllm 现在支持统一的模型注册系统，允许根据配置自动选择和加载不同的模型架构。

## 文件结构

```
nanovllm/
├── models/
│   ├── __init__.py          # 导出模型注册功能
│   ├── model_registry.py    # 核心模型注册系统
│   ├── cpm4.py             # MiniCPM/CPM4 模型实现
│   └── qwen3.py            # Qwen3 模型实现
└── engine/
    └── model_runner.py     # 已修改为使用模型注册系统
```

## 支持的模型

当前支持的模型架构和别名：

| 架构名称 | 别名 | 模型类 |
|---------|------|--------|
| `MiniCPMForCausalLM` | `minicpm`, `cpm4`, `MiniCPM` | `Cpm4ForCausalLM` |
| `Qwen3ForCausalLM` | `qwen3`, `Qwen3` | `Qwen3ForCausalLM` |

## 使用方法

### 1. 自动模型选择

修改后的 `model_runner.py` 会自动根据 HuggingFace 配置选择合适的模型：

```python
from nanovllm.models import create_model

# 在 ModelRunner.__init__ 中
self.model = create_model(hf_config)
```

模型选择优先级：
1. 首先检查 `config.architectures[0]`
2. 如果失败，检查 `config.model_type`
3. 如果还是失败，根据配置类名推断

### 2. 手动模型选择

```python
from nanovllm.models import get_model_class

# 根据架构名称获取模型类
model_class = get_model_class("MiniCPMForCausalLM")
model = model_class(hf_config)

# 或者使用别名
model_class = get_model_class("qwen3")
model = model_class(hf_config)
```

### 3. 注册新模型

```python
from nanovllm.models import register_model

class MyCustomModel:
    def __init__(self, config):
        # 模型初始化逻辑
        pass

# 注册新模型
register_model("MyCustomForCausalLM", MyCustomModel)
register_model("custom", MyCustomModel)  # 别名

# 现在可以使用了
model_class = get_model_class("custom")
```

### 4. 列出支持的模型

```python
from nanovllm.models import list_supported_models

models = list_supported_models()
for name, model_class in models.items():
    print(f"{name}: {model_class.__name__}")
```

## 配置要求

确保你的模型配置文件（`config.json`）包含以下字段之一：

```json
{
  "architectures": ["MiniCPMForCausalLM"],
  // 或者
  "model_type": "minicpm"
}
```

## 错误处理

如果模型无法识别，系统会抛出详细的错误信息：

```
ValueError: Cannot determine model type from config. 
Config architecture: None, model_type: unknown, config_class: UnknownConfig. 
Available models: ['MiniCPMForCausalLM', 'Qwen3ForCausalLM', 'minicpm', 'qwen3', 'cpm4', 'MiniCPM', 'Qwen3']
```

## 测试

运行测试脚本验证功能：

```bash
cd /path/to/nano-vllm
python test_model_registry.py
```

## 扩展性

添加新模型只需要：

1. 实现模型类（继承适当的基类）
2. 在 `model_registry.py` 的 `_register_default_models()` 方法中注册
3. 或者在运行时使用 `register_model()` 函数动态注册

这种设计使得添加新模型架构变得非常简单，同时保持了向后兼容性。
