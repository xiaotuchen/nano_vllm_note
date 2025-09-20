# nano-vllm cookbook
这里主要是讲解nano-vllm从而更好了解推理引擎，主要从三个方面讲解，一个是入口函数的调用逻辑，一个是推理引擎的核心组成，包括引擎的核心组成模块以及算子层实现。

在原有的功能上适配了MiniCPM4，并且增加了注册新模型的功能，在nano_vllm/models/cpm4.py和nano_vllm/models/registry.py文件
## 入口函数调用逻辑
从入口函数来看首先会调用LLM方法启动引擎，然后调用generate方法。
``` python
def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)
```

### 1首先需要初始化引擎：LLM（）方法
主要逻辑是：
    1. 根据模型并行数进行多线程操作，（nano-vllm仅支持模型并行）
    2. 启动模型推理器，这里需要在后文中详细展开
    3. 加载分词器。
    4. 初始化调度器，这里展开将。
``` python
# llm_engine.py
class LLMEngine:
    """
    LLMEngine类负责管理整个推理流程，包括模型初始化、请求管理、调度、分布式并行、tokenizer处理等。
    """

    def __init__(self, model, **kwargs):
        # 获取Config类的所有字段名
        config_fields = {field.name for field in fields(Config)}
        # 过滤kwargs，只保留Config需要的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 初始化配置对象
        config = Config(model, **config_kwargs)
        # 存储分布式进程和事件
        self.ps = []
        self.events = []
        # 使用spawn方式创建多进程上下文，逐哟啊是考虑多进程
        ctx = mp.get_context("spawn")
        # 启动tensor parallel的worker进程（主进程为0号，worker从1开始）
        # tp的多进程管理
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建一个进程间同步事件，用于主进程和worker进程之间的通信与同步
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 创建一个新的worker进程，目标函数是ModelRunner，参数包括配置、进程编号i、同步事件
            process.start()  # 启动该worker进程，让其在后台运行
            self.ps.append(process)  # 将进程对象保存到进程列表，便于后续管理和回收
            self.events.append(event)  # 将事件对象保存到事件列表，便于主进程与各worker通信
        # 主进程的ModelRunner实例
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 设置终止token id
        config.eos = self.tokenizer.eos_token_id
        # 初始化调度器
        self.scheduler = Scheduler(config)
        # 注册退出时的清理函数
        atexit.register(self.exit)
```
上初始化引擎第二步：初始化模型推理器操作，核心操作主要包括以下几点：
1. 加载模型类以及权重
2. 预热模型，根据最大的批次空数据跑一次prefill。
3. 按照本文内存管理层级中的KV-cache分配，先将内存分配给每一层的k和v
4. 如果不使用eager模式，保存多个不同batch的计算图，用于后面decode减速，基本原理是可以在后面decode阶段直接换保存的计算图中的数据，从而提速，即使用cudagraph
5. 多进程之间进行同步。
``` python
# model_runner.py
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    # 初始化分布式进程组，使用NCCL后端和TCP通信
    dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
    torch.cuda.set_device(rank)  # 设置当前进程的GPU
    default_dtype = torch.get_default_dtype()  # 记录默认数据类型
    torch.set_default_dtype(hf_config.torch_dtype)  # 设置模型数据类型
    torch.set_default_device("cuda")  # 设置默认设备为cuda
    self.model = Qwen3ForCausalLM(hf_config)  # 构建模型
    load_model(self.model, config.model)  # 加载权重
    self.sampler = Sampler()  # 构建采样器
    self.warmup_model()  # 预热模型，用尽量大的batch和token计算一次
    self.allocate_kv_cache()  # 分配KV缓存，计算可用的KV缓存块数量，并且绑定到每个层的k和v
    if not self.enforce_eager:
        self.capture_cudagraph()  # 捕获CUDA图以加速推理，原理是预先加载了计算图，但是要预存计算图会占用部分的显存
    torch.set_default_device("cpu")  # 恢复默认设备为cpu
    torch.set_default_dtype(default_dtype)  # 恢复默认数据类型

    # 多进程共享内存和同步
    if self.world_size > 1:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 主进程创建共享内存
            dist.barrier()  # 同步. rank 0 reaches dist.barrier(), it pauses and waits.
        else:
            dist.barrier()  # 等待主进程 rank i reaches dist.barrier(), it pauses and waits. Once both GPU rank 0 and other rank i have called barrier(), both are released simultaneously to proceed to the next step.
            self.shm = SharedMemory(name="nanovllm")  # 其他进程连接共享内存
            self.loop()  # 子进程进入循环等待任务
```
初始化引擎中第4步，初始化调度器：
1. 初始化以block为单位的kvcache的内存管理器
2. 设置两个队列，一个是等待队列，等待队列中都是还没开始运行的请求，一个是运行队列，其中都是至少在进行prefilling的请求，队列中其实用来管理发过来的请求。
``` python
class Scheduler:
    """
    Scheduler类负责管理序列的调度，包括等待队列、运行队列、KV缓存块的分配与回收。
    """

    def __init__(self, config: Config):
        # 最大并发序列数
        self.max_num_seqs = config.max_num_seqs
        # 最大批处理token数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 终止token id
        self.eos = config.eos
        # KV缓存块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待队列
        self.waiting: deque[Sequence] = deque()
        # 运行队列
        self.running: deque[Sequence] = deque()
```
### 使用引擎的generate方法：llm.generate()
1. 首先将generate方法中的所有request添加到调度器中的等待队列中
2. 当等待队列和运行队列中仍不为空时，始终进行step（），step逻辑主要如下：
  - 1. 调度器运行调度
        1. 如果等待队列中不为空，优先将等待序列中的request进行prefilling
        2. 否则将按运行队列中的顺序进行decode
  - 2. 对于已经生成结束符eos_token或者达到最大生成长度的request，解码成文字，并且从队列中删除
  - 3. 按照prefilling和decode分别统计token的吞吐量
3. 将完成的request的输出output按照输入的顺序进行排序。
4. 返回结果
``` python
def step(self):
    """
    执行一步推理，包括调度、模型运行、后处理，返回输出和token数。
    如果是prefill阶段，返回本轮所有序列的token总数；如果是decode阶段，那每个seq仅生成一个token。
    """
    seqs, is_prefill = self.scheduler.schedule()  # 调度本轮要推理的序列，并判断是否为prefill阶段（True为prefill，False为decode）
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # 调用模型推理，输入为本轮序列和阶段类型，返回生成的token id
    self.scheduler.postprocess(seqs, token_ids)  # 对推理结果做后处理，如判断哪些序列已完成、更新状态等，如是否达到最大长度或者结束符，判断结束
    # 收集本轮已完成的序列输出（只收集已完成的序列，包含序列id和生成的token id列表）
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    # 统计token数：如果是prefill阶段，统计本轮所有序列的token总数；如果是decode阶段，统计本轮decode的序列数（取负号用于区分）
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens  # 返回已完成序列的输出和本轮token统计
def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成接口，支持进度条显示，返回解码后的文本和token id。
        """
     
        # 添加所有请求到调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 主推理循环，直到所有序列完成
        while not self.is_finished(): # 是否结束由调度器判断
            t = perf_counter()# 高精度计时
            output, num_tokens = self.step()
            # 更新进度条和吞吐率
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # 按序列id排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        # 解码为文本，并保留token id
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
```
## 核心模块树状结构
```
nano-vllm/
├── nanovllm/
│   ├── __init__.py          # 包导入入口
│   ├── llm.py               # LLM主类，推理API入口
│   ├── sampling_params.py   # 采样参数定义
│   ├── config.py            # 配置参数定义
│   ├── engine/#这个文件夹下都是功能文件（调度层）
│   │   ├── llm_engine.py    # 推理主流程，API接口
│   │   ├── model_runner.py  # 模型加载与分布式推理
│   │   ├── scheduler.py     # 推理调度与分块分配
│   │   ├── sequence.py      # 输入序列对象
│   │   ├── block_manager.py # KV缓存分块管理
│   ├── layers/#这个文件夹下都是算子文件（算子层）
│   │   ├── linear.py        # 线性层（并行/分片等）
│   │   ├── attention.py     # 注意力层与KV缓存写入
│   │   ├── activation.py    # 激活值
│   │   ├── layernorm.py     # 层归一化
│   │   ├── linear.py        # 线性层
│   │   ├── rotary_embedding.py # rope实现
│   │   ├── embed_head.py。  # embed_head.py层
│   ├── models/# 适配接口层，比如要新适配模型就在这里定义
│   │   ├── qwen3.py         # Qwen3模型结构定义
│   ├── utils/
│   │   ├── context.py       # 推理上下文管理
│   │   ├── loader.py        # 权重加载工具
│   └── ...                  # 其他辅助模块
```
## 主要功能模块
### 主引擎-llm_engine.py
1. 主要作用
- 负责管理整个大模型推理流程，包括模型初始化、分布式并行、请求管理、调度、分词器处理、资源清理等。
- 封装了多进程（Tensor Parallel）推理、批量请求、吞吐统计等高阶功能，提供统一的推理接口。

---
2. 主要成员变量
- ps：存储所有分布式 worker 进程对象（除主进程外）。
- events：存储主进程与各 worker 进程的同步事件对象。
- model_runner：主进程的 ModelRunner 实例，负责实际的模型推理。
- tokenizer：transformers 的分词器，用于文本与 token id 的互转。
- scheduler：推理调度器，负责序列的调度、分配、回收等。
- Config：推理配置对象，包含模型路径、并行数、显存利用率等参数。

---
3. 初始化流程（__init__）
1. 参数过滤与配置初始化
  - 只保留传入参数中属于 Config 的字段，构造 Config 对象。
2. 多进程环境准备
  - 使用 spawn 方式创建多进程上下文，适配多卡并行。
3. Tensor Parallel 进程启动
  - 除主进程外，为每个并行卡创建一个 worker 进程（ModelRunner），并通过事件对象实现主从同步。
  - 主进程 rank=0，worker 进程 rank=1~N。
4. 主进程 ModelRunner 初始化
  - 主进程自身也创建一个 ModelRunner 实例，负责本地推理。
5. 分词器加载
  - 加载 transformers 分词器，并设置终止 token id。
6. 调度器初始化
  - 创建 Scheduler 实例，管理推理任务的调度与资源分配。
7. 注册退出清理函数
  - 程序退出时自动清理所有进程和资源。

---
4. 主要方法
4.1 exit
- 清理资源，关闭所有进程和模型 runner，确保进程安全退出。
4.2 add_request
- 添加推理请求，将 prompt 编码为 token id 并加入调度队列，支持字符串和 token id 两种输入。
4.3 step
- 执行一步推理，包括调度、模型运行、后处理，返回已完成序列的输出和本轮 token 统计。
  - 优先waiting队列中的seq进行转移到runing队列，并且对这些request进行prefilling，再一次step中即可完成prefilling
  - 当waiting中没有seq时，则对runing队列中的seq进行decode，在一次step中每个seq仅生成一个token，对于新生成的token需要考虑是否申请新的block进行储存。
- 支持 prefill 阶段（批量填充）和 decode 阶段（增量生成）的区分与吞吐统计。
4.4 is_finished
- 判断所有序列是否推理完成（由调度器判断，是否有结束符或者达到最大程度）。
4.5 generate
- 批量生成接口，支持进度条显示，自动处理多个 prompt 和采样参数。
- 主循环不断调用 step，直到所有序列完成，期间统计 prefill/decode 吞吐率并实时展示。
- 最终输出按序列 id 排序，返回解码文本和 token id。

---
5. 分布式与并行
- 采用多进程（Tensor Parallel）方式，每个进程负责一部分模型权重和推理任务，极大提升大模型推理吞吐。
- 主进程与 worker 进程通过事件和共享内存通信，保证任务同步和资源高效利用。

---
6. 资源管理与清理
- 通过 atexit 注册退出清理函数，确保所有进程和资源在程序结束时被正确释放，防止资源泄漏。
内存管理层级-block_manager.py
内存管理逻辑总结

### 内存管理层级-block_manager.py
#### 内存管理逻辑总结
1. Block 类
- 表示 KV 缓存中的一个块（block），包含：
  - block_id：唯一编号
  - ref_count：引用计数，表示被多少序列引用
  - hash：块内容的哈希值，用于缓存查找和复用
  - token_ids：当前块存储的 token id 列表
- 提供 update 和 reset 方法，分别用于更新块内容和重置状态
2. BlockManager 类
主要成员
- block_size：每个 block 能存储的 token 数
- blocks：所有 Block 对象的列表
- hash_to_block_id：哈希到 block_id 的映射，用于缓存查找和复用
- free_block_ids：空闲 block 的 id 队列
- used_block_ids：已分配 block 的 id 集合
主要方法
2.1 分配与回收
- allocate(seq)：为一个序列分配所需的 block，支持缓存复用
  - 优先查找哈希表是否有可复用的 block，命中则复用，否则分配新 block
  - 分配时重置 block 状态，引用计数设为 1，并加入 used_block_ids
- deallocate(seq)：回收一个序列占用的所有 block
  - 遍历 block_table，将每个 block 的引用计数减 1
  - 如果引用计数为 0，则回收该 block，放回 free_block_ids 队列
2.2 追加 token 时的动态管理
- can_append(seq)：判断是否可以为序列追加一个 block
  - 当 len(seq) % block_size == 1 时，说明刚新开一个 block，需要有空闲 block 可用
- may_append(seq)：处理序列追加 token 时的 block 分配和哈希更新逻辑
  - 情况1：len(seq) % block_size == 1，新开 block，分配新 block 并加入 block_table
  - 情况2：len(seq) % block_size == 0，刚好填满 block，计算 hash，注册到哈希表
  - 情况3：其它情况，继续往当前 block 追加 token，无需分配新 bloc
2.3 哈希与复用

- compute_hash(token_ids, prefix)：计算一组 token_ids 的哈希值（可选带前缀），用于缓存查找和复用
- 每个 block 的内容和前缀 hash 计算出唯一哈希值，作为缓存查找和复用的依据

---
3. 管理策略总结
- BlockManager 通过 block 切分、哈希查找、引用计数和空闲队列，实现了高效的 KV cache 显存分配、回收和复用
- 支持 LLM 推理过程中的动态缓存管理和高吞吐推理
- 主要目标是节省显存、加速推理、支持缓存复用
#### KV-cache分配
- 1在vllm中最小内存管理单元是block，一个block默认是256个token占用的内存，每个block占用的内存计算如下：
```
block_memory=2 * num_hidden_layers * block_size * num_kv_heads * head_dim *torch_dtype_itemsize
```
这里计算的是如果输入一个block，在前向传播中每一层占用的kvcache之和。
- 2在初始化引擎的过程中，将计算总共能分配的显存：
```
num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```
基本上可以理解为将gpu内存乘以最大接受使用率减去已经用过的内存得到还可以用的内存，将这个内存除以每个block占用的字节数就可以获得能够分配出多少个block
- 3预先分配每层的kv_cache显存：
将第2步计算出来的num_kvcache_blocks分配到每一个层的k和v去。每层的k和v所需要的cache内存如下：
```
k_cache_per_layer=num_kvcache_blocks*block_size*num_kv_heads*head_dim
```

### 调度逻辑总结-scheduler.py
#### 主要作用
- 统一管理 LLM 推理任务的调度流程，包括序列的等待、运行、抢占、KV 缓存分配与回收等。
- 保证显存资源高效利用、吞吐最大化、序列公平调度。

---
#### 主要成员变量
- waiting：等待队列，存放待处理的新序列。
- running：运行队列，存放正在推理的序列。
- block_manager：KV 缓存块管理器，负责 block 的分配与回收。
- max_num_seqs：最大并发序列数。
- max_num_batched_tokens：最大批处理 token 数。
- eos：终止 token id。

---
#### 核心方法
is_finished
- 判断所有序列是否都已完成（等待和运行队列均为空）。
add
- 新增一个序列到等待队列。
schedule
- 调度方法，分为 prefill 和 decode 两个阶段，返回本轮调度的序列列表和是否为 prefill 阶段。
- prefill 阶段：优先从 waiting 队列调度新序列，分配 KV 块，直到达到最大并发数或 token 数限制或者请求不足。
- decode 阶段：对 running 队列中的序列做增量生成（每轮每个序列生成一个 token），如 KV 块不足则抢占其他序列资源。
preempt
- 抢占一个序列，将其状态设为 WAITING 并回收 KV 块，重新加入等待队列。
postprocess
- 后处理方法，将生成的 token 追加到序列，并判断是否终止（eos 或达到最大 token 数）。
- 如果终止则回收 KV 块并移出运行队列。

---
#### 调度流程总结
prefill 阶段  
  - 从 waiting 队列调度新序列，分配 KV 块，加入 running 队列。
  - 达到并发/显存上限或无可调度序列后，进入 decode 阶段。
decode 阶段  
  - 对 running 队列中的序列增量生成 token。
  - 如 KV 块不足，优先抢占 running 队列末尾的序列，回收其资源。
  - 追加 token 时动态分配 block，保证显存利用最大化。
后处理  
  - 每轮推理后，将生成的 token 追加到序列。
  - 判断是否遇到终止符（eos）或达到最大 token 数，若是则回收资源并移出运行队列。

### 模型执行 - ModelRunner.py

#### 1. 主要作用
- 负责单个分布式进程（或主进程）上的模型加载、KV缓存分配、推理执行、CUDA Graph 捕获、进程间通信等核心功能。
- 支持分布式 tensor parallel、CUDA Graph 加速、KV cache 动态分配与复用。

#### 2. 主要成员变量
- config：推理配置对象。
- block_size：KV 缓存块大小。
- enforce_eager：是否强制使用 eager（非 CUDA Graph）模式。
- world_size：分布式进程总数（通常等于总 GPU 数）。
- rank：当前进程编号。
- event：进程间同步事件（用于多进程通信）。
- model：实际的 LLM 模型实例。
- sampler：采样器，用于从 logits 采样 token。
- kv_cache：KV 缓存张量。
- graph_bs、graphs、graph_vars：CUDA Graph 相关变量（用于 decode 加速）。
- shm：共享内存对象（多进程通信用）。

#### 3. 初始化流程（__init__）
##### 分布式初始化
   - 通过 dist.init_process_group 初始化 NCCL 通信组，设置当前进程的 GPU。
   - 配置 world_size（进程总数）和 rank（当前进程编号）。
##### 模型与采样器加载
   - 设置默认 dtype 和 device 为 CUDA，加载 Qwen3ForCausalLM 模型。
   - 通过 load_model 加载预训练权重，初始化 Sampler 采样器。
##### 模型预热与 KV 缓存分配
   - 调用 warmup_model 预热模型，减少首次推理延迟。
   - 调用 allocate_kv_cache 动态分配并绑定 KV 缓存到各层。
##### CUDA Graph 捕获
   - 如果不是 enforce_eager 模式，调用 capture_cudagraph 捕获不同 batch size 的 CUDA Graph。
##### 恢复默认设置
   - 恢复默认 device 为 CPU 和原始 dtype，避免影响后续代码。
##### 多进程共享内存与同步
   - 主进程创建共享内存，子进程连接并进入 loop 循环等待任务。

#### 4. 核心方法详解

##### 4.1 exit
- 作用：安全释放所有资源。
- 实现：
  - 关闭共享内存，主进程负责删除共享内存。
  - 删除 CUDA Graph 相关资源。
  - 同步所有 CUDA 操作，销毁分布式进程组。

##### 4.2 loop
- 作用：子进程循环等待主进程任务。
- 实现：
  - 通过 read_shm 从共享内存读取任务。
  - 调用 call 执行任务，直到收到 "exit" 命令退出。

##### 4.3 read_shm / write_shm
- 作用：进程间通过共享内存读写任务。
- 实现：
  - read_shm：子进程等待事件触发，读取序列化数据并反序列化。
  - write_shm：主进程序列化任务数据，写入共享内存并通知所有子进程。

##### 4.4 call
- 作用：统一的方法调用接口，支持分布式调用。
- 实现：
  - 主进程通过 write_shm 分发任务到子进程。
  - 通过反射获取方法并执行，返回结果。

##### 4.5 warmup_model
- 作用：预热模型，减少首次推理延迟和显存碎片。
- 实现：
  - 清空 CUDA 缓存，重置显存统计。
  - 构造最大规模的虚拟序列，执行一次 prefill 推理。
  - 再次清空缓存，确保后续推理环境干净。

##### 4.6 allocate_kv_cache
- 作用：动态分配 KV 缓存张量，并绑定到模型每一层。
- 实现：
  - 查询 GPU 显存信息，计算可用显存。
  - 根据模型层数、头数、维度等计算单个 block 字节数。
  - 分配形状为 [2, 层数, 块数, 块大小, 头数, 头维度] 的 KV 缓存张量。
  - 遍历模型所有层，将 K/V 缓存绑定到对应层的 k_cache 和 v_cache 属性。

##### 4.7 prepare_block_tables
- 作用：构造 block_table 张量，补齐长度便于 batch 处理。
- 实现：
  - 找到所有序列中最长的 block_table 长度。
  - 用 -1 填充较短的 block_table，转为 CUDA 张量。

##### 4.8 prepare_prefill
- 作用：构造 prefill 阶段的输入张量和推理上下文。
- 实现：
  - 收集所有序列未缓存的 token ids 和位置信息。
  - 计算 query/key 的累加长度（cu_seqlens）和最大长度。
  - 生成 slot_mapping，指示每个 token 在 KV cache 中的物理位置。
  - 如有 prefix cache，构造 block_tables。
  - 设置推理上下文，返回 input_ids 和 positions。

##### 4.9 prepare_decode
- 作用：构造 decode 阶段的输入张量和推理上下文。
- 实现：
  - 收集每个序列的最后一个 token 和当前位置。
  - 计算每个 token 在 KV cache 中的 slot 位置。
  - 构造 context_lens 和 block_tables。
  - 设置 decode 推理上下文。

##### 4.10 prepare_sample
- 作用：构造采样温度张量。
- 实现：
  - 从所有序列中提取温度参数。
  - 转为 CUDA 张量，用于后续采样。

##### 4.11 run_model
- 作用：执行模型推理，支持 prefill/decode 和 CUDA Graph 加速。
- 实现：
  - 条件判断：如果是 prefill、enforce_eager 或 batch size > 512，使用常规推理。
  - 常规推理：直接调用 model 前向传播并计算 logits。
  - CUDA Graph 推理，eager模式不会使用CUDA Graph：
    - 根据 batch size 选择合适的预捕获图。
    - 清零图变量，填入当前推理数据。
    - 调用 graph.replay() 重放推理图。
    - 从输出张量提取 logits。

##### 4.12 run
- 作用：推理统一入口，执行一次完整推理并采样。
- 实现：
  - 根据 is_prefill 调用相应的 prepare 方法，这里判断走prefilling还是走decode。
  - 主进程准备采样温度，调用 run_model 执行推理。
  - 主进程进行采样，重置推理上下文。

##### 4.13 capture_cudagraph
- 作用：捕获不同 batch size 的 CUDA Graph，加速 decode 阶段推理。
- 实现：
  - 预分配所有推理相关张量（input_ids、positions、outputs 等）。
  - 定义支持的 batch size 列表 [1, 2, 4, 8, 16, 32, ...]。
  - 从大到小遍历每个 batch size：
    - 创建 CUDA Graph 对象。
    - 设置推理上下文，执行一次 warmup。
    - 用 torch.cuda.graph 捕获推理过程。
    - 保存图对象和图池。
  - 保存所有图变量供后续复用。

### 序列管理 - Sequence.py

#### 1. 主要作用
- 表示推理过程中的单个序列对象，封装了序列的状态、token管理、block分配、采样参数等信息。
- 支持序列的动态扩展、缓存管理、状态跟踪和序列化/反序列化。

#### 2. SequenceStatus 枚举
- WAITING：序列等待调度状态。
- RUNNING：序列正在推理状态。
- FINISHED：序列推理完成状态。

#### 3. 主要成员变量

##### 3.1 类级别变量
- block_size：每个 block 的 token 数量（默认 256）。
- counter：全局自增序列号生成器。

##### 3.2 实例变量
- seq_id：唯一序列 ID。
- status：当前序列状态（WAITING/RUNNING/FINISHED）。
- token_ids：完整的 token id 列表（包含 prompt 和 completion）。
- last_token：最后一个 token id。
- num_tokens：当前 token 总数。
- num_prompt_tokens：prompt 部分 token 数。
- num_cached_tokens：已缓存的 token 数。
- block_table：记录分配的 block id 列表。
- temperature、max_tokens、ignore_eos：采样参数。

#### 4. 核心方法详解

##### 4.1 __init__
- 作用：初始化序列对象。
- 实现：
  - 分配唯一序列 ID，设置初始状态为 WAITING。
  - 复制输入 token_ids，记录 prompt token 数。
  - 从 SamplingParams 提取采样参数。
  - 初始化缓存计数和 block_table用来记录被使用的block_id。

##### 4.2 __len__
- 作用：返回当前序列的 token 总数。
- 实现：直接返回 self.num_tokens。

##### 4.3 __getitem__
- 作用：支持下标访问序列中的 token。
- 实现：返回 self.token_ids[key]。

##### 4.4 属性方法（Properties）

###### is_finished
- 作用：判断序列是否已完成。
- 实现：检查状态是否为 FINISHED。

###### num_completion_tokens
- 作用：返回已生成的 completion token 数。
- 实现：num_tokens - num_prompt_tokens。

###### prompt_token_ids
- 作用：返回 prompt 部分的 token id 列表。
- 实现：token_ids[:num_prompt_tokens]。

###### completion_token_ids
- 作用：返回 completion 部分的 token id 列表。
- 实现：token_ids[num_prompt_tokens:]。

###### num_cached_blocks
- 作用：返回已缓存的 block 数。
- 实现：num_cached_tokens // block_size。

###### num_blocks
- 作用：返回当前序列需要的 block 总数（向上取整）。
- 实现：(num_tokens + block_size - 1) // block_size。

###### last_block_num_tokens
- 作用：返回最后一个 block 实际包含的 token 数。
- 实现：num_tokens - (num_blocks - 1) * block_size。

##### 4.5 block
- 作用：获取第 i 个 block 的 token id 列表。
- 实现：
  - 检查索引范围有效性。
  - 返回 token_ids[i*block_size: (i+1)*block_size]。

##### 4.6 append_token
- 作用：追加一个新生成的 token。
- 实现：
  - 将 token_id 添加到 token_ids 列表末尾。
  - 更新 last_token 和 num_tokens。

##### 4.7 序列化方法

###### __getstate__
- 作用：序列化序列对象，用于进程间通信。
- 实现：
  - 返回关键信息元组：(num_tokens, num_prompt_tokens, num_cached_tokens, block_table, token_data)。
  - 如果没有 completion tokens，保存完整 token_ids；否则只保存 last_token。

###### __setstate__
- 作用：反序列化恢复序列对象状态。
- 实现：
  - 从状态元组恢复各个字段。
  - 根据是否有 completion tokens 决定恢复 token_ids 还是 last_token。

#### 5. 设计特点

##### 5.1 内存优化
- 通过 block 机制管理大序列，避免连续内存分配。
- 序列化时根据情况选择保存完整 token_ids 或仅保存 last_token，减少传输开销。

##### 5.2 状态管理
- 清晰的状态转换：WAITING → RUNNING → FINISHED。
- 分离 prompt 和 completion tokens，便于不同阶段的处理。

##### 5.3 缓存支持
- num_cached_tokens 和 num_cached_blocks 支持 KV cache 复用。
- block_table 记录物理 block 分配，支持高效的缓存管理。

##### 5.4 采样参数集成
- 直接集成温度、最大 token 数、EOS 忽略等采样参数。
- 便于后续采样和终止条件判断。

#### 6. 总结
Sequence 类是 LLM 推理中序列管理的核心数据结构，封装了序列的完整生命周期管理，支持动态扩展、缓存复用、状态跟踪和高效的进程间通信，为高性能批量推理提供了坚实的基础。

## 主要算子模块
### 注意力算子-Attention.py
### Triton编写的store_kvcache方法
slot_map是用于vllm的page_attention的核心组件，这样可以将一个变量的内存放在上非连续内存用slot_map进行记录和管理，
所以以下triton函数核心功能是读取连续的key、value显存，并且将其储存到slot_map上给出的非连续内存地址，这样需要频繁的内存访问，所以使用triton进行编写加速。
``` triton
@triton.jit
def store_kvcache_kernel(
    key_ptr,              # key 张量的指针
    key_stride,           # key 张量第一个维度的跨度（每个 token 的步长）
    value_ptr,            # value 张量的指针
    value_stride,         # value 张量第一个维度的跨度
    k_cache_ptr,          # 全局 k_cache 张量的指针
    v_cache_ptr,          # 全局 v_cache 张量的指针
    slot_mapping_ptr,     # slot_mapping 张量的指针（每个 token 写入 KV cache 的物理位置）
    D: tl.constexpr,      # 每个 token 的 KV 向量长度（num_heads * head_dim）
):
    idx = tl.program_id(0)  # 当前线程处理的 token 索引
    key_offsets = idx * key_stride + tl.arange(0, D)      # 计算当前 token 的 key 数据在 key 张量中的偏移
    value_offsets = idx * value_stride + tl.arange(0, D)  # 计算当前 token 的 value 数据在 value 张量中的偏移
    key = tl.load(key_ptr + key_offsets)                  # 加载当前 token 的 key 向量
    value = tl.load(value_ptr + value_offsets)            # 加载当前 token 的 value 向量
    slot = tl.load(slot_mapping_ptr + idx)                # 读取当前 token 应写入 KV cache 的 slot 位置
    cache_offsets = slot * D + tl.arange(0, D)            # 计算当前 token 在全局 KV cache 中的写入偏移
    tl.store(k_cache_ptr + cache_offsets, key)            # 将 key 向量写入全局 k_cache 的对应 slot
    tl.store(v_cache_ptr + cache_offsets, value)          # 将 value 向量写入全局 v_cache 的对应 slot


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1         # 保证最后一维连续
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 保证 head 维度步长正确
    assert k_cache.stride(1) == D and v_cache.stride(1) == D     # 保证 KV cache 步长正确
    assert slot_mapping.numel() == N                             # slot_mapping 数量等于 token 数
    store_kvcache_kernel[(N,)](                                  
        key, key.stride(0),                                      # 传入 key 张量和步长
        value, value.stride(0),                                  # 传入 value 张量和步长
        k_cache, v_cache,                                        # 传入全局 KV cache
        slot_mapping, D                                          # 传入 slot_mapping 和每个 token 的 KV 向量长度
    )
```
#### 注意力机制计算-flashattention
没有什么好讲的，prefilling和decode都是直接调用的flash-attention的flash_attn_varlen_func方法和flash_attn_with_kvcache方法

### embeding和llm_head-embed_head.py
#### embeding的分布式
1. 原理如图所示，将embeding层按照token拆成TP份
2. 将当前设备embeding的位置不存在的X不参与后续计算（注意代码实现与图不同，因为如果将weight置零有计算开销和储存成本）
3. 将每个设备的embeding层和对应的X进行计算，并且按第2步忽略X的某些位置。
4. 将每个设备的embeding层通过allreduce进行聚合

``` python
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()  # 当前进程的分布式 rank
        self.tp_size = dist.get_world_size()  # 分布式总进程数（总卡数）
        assert num_embeddings % self.tp_size == 0  # 保证词表能被均匀分片
        self.num_embeddings = num_embeddings  # 总词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size  # 每张卡分到的词表大小
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank  # 本卡负责的词表起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition  # 本卡负责的词表结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # 本卡的 embedding 参数
        self.weight.weight_loader = self.weight_loader  # 绑定权重加载函数

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data  # 获取本地参数数据
        shard_size = param_data.size(0)  # 本卡分片的大小
        start_idx = self.tp_rank * shard_size  # 本卡分片在全局权重中的起始位置
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)  # 截取本卡负责的权重分片
        assert param_data.size() == loaded_weight.size()  # 检查分片尺寸一致
        param_data.copy_(loaded_weight)  # 拷贝权重到本地参数

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)  # 只保留本卡负责的 token
            x = mask * (x - self.vocab_start_idx)  # 将本卡负责的 token 索引映射到本地 embedding 表
        y = F.embedding(x, self.weight)  # 查表得到 embedding
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # 非本卡负责的 token embedding 置零
            dist.all_reduce(y)  # 多卡间 embedding 求和，聚合所有卡的结果
        return y  # 返回最终 embedding
```

#### lm_head的分布式
就是普通的列并行这个没什么好说的

### 线性层算子-linear.py
在这个文件下实现了行并行和列并行，以及q、k、v的投影计算都是使用的列并行。

比较值得一说的是**合并列并行**（就是某一层的activation平行输入给多个weight作乘法，可以将这多个weight合并成一个weight进行列并行，再拆分，这样的话本来多次列并行编程一次列并行，减少了kernel启动和通信时间）：
```python
class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```
## 适配模型qwen3.py
### qwen3-attention
1. q、k、v矩阵投影使用的是单独的列并行
2. 直接调用上述的Attention模块进行注意力计算
3. **o的投影矩阵采用了行并行，这里是因为连续的两个线性层中，第一个层使用列并行，第二个行采用行并行，中间可以不进行额外的通信**
```python
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output
```
### qwen3-mlp
qwen3的mlp层一共有三个投影矩阵，$$W_{gate}，W_{up}，W_{down}$$,计算流程如下：
$$g, u = x W_{gate}^T，x W_{up}^T$$
$$y = \mathrm{SiLU}(g) \odot u$$
$$z = y W_{down}^T$$
nano_vllm的代码实现是将$$W_{gate}，W_{up}$$合并在一起做列并行，即
$$[g, u] = x W_{gu}^T$$
并且将SiLU函数隐藏到了SiluAndMul()函数中
```python
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```
