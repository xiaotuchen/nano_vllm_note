import pickle
'''serializes a Python object into a bytes object, allowing the object to be stored in memory, 
sent over a network, or saved for later reconstruction without writing directly to a file. 
pickle.dump(), which writes directly to a file object, 
pickle.dumps() is used when file I/O isn’t wanted, such as for database or network operations.
'''
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
'''The multiprocessing.shared_memory.SharedMemory class in Python provides a way to 
create and manage blocks of memory that can be shared between multiple processes directly, 
without the need for copying data through inter-process communication like pipes or queues.
'''

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config  # 保存配置对象
        hf_config = config.hf_config  # HuggingFace模型配置
        self.block_size = config.kvcache_block_size  # KV缓存块大小
        self.enforce_eager = config.enforce_eager  # 是否强制eager模式
        self.world_size = config.tensor_parallel_size  # 并行进程总数
        self.rank = rank  # 当前进程rank
        self.event = event  # 进程间同步事件

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
                dist.barrier()  # 等待其它进程，同步
            else:
                dist.barrier()  # 等待主进程
                self.shm = SharedMemory(name="nanovllm")  # 其他进程连接共享内存
                self.loop()  # 子进程进入循环等待任务

    def exit(self):
        # 释放资源
        if self.world_size > 1:
            self.shm.close()  # 关闭共享内存
            dist.barrier()  # 同步
            if self.rank == 0:
                self.shm.unlink()  # 主进程删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 删除CUDA图相关资源
        torch.cuda.synchronize()  # 等待所有CUDA操作完成
        dist.destroy_process_group()  # 销毁分布式进程组

    def loop(self):
        # 子进程循环读取共享内存，等待主进程任务
        while True:
            method_name, args = self.read_shm()  # 从共享内存读取任务
            self.call(method_name, *args)  # 执行任务
            if method_name == "exit":
                break  # 收到退出命令则退出循环

    def read_shm(self):
        # 从共享内存读取任务（子进程用）
        assert self.world_size > 1 and self.rank
        self.event.wait()  # 等待事件触发
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化任务
        self.event.clear()  # 清除事件 Now event.is_set() is False, causing threads that call event.wait() to block until event.set() is called again.
        return method_name, args

    def write_shm(self, method_name, *args):
        # 向共享内存写入任务（主进程用）
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])  # 序列化任务 
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度
        self.shm.buf[4:n+4] = data  # 写入数据
        for event in self.event:
            event.set()  # 通知所有子进程 Now event.is_set() is True, awakening all threads waiting for that event object.

    def call(self, method_name, *args):
        # 调用指定方法
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 主进程写任务到共享内存
        method = getattr(self, method_name, None)  # 获取方法 getattr to retrieve the object "self's" attribute "method_name", cannot find then None 
        return method(*args)  # 执行方法

    def warmup_model(self):
        # 预热模型，减少首次推理延迟
        torch.cuda.empty_cache()           # 清空未使用的CUDA显存缓存，释放GPU内存
        torch.cuda.reset_peak_memory_stats() # 重置CUDA的显存峰值统计，便于后续准确监控显存使用峰值
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]  # 构造填充序列
        self.run(seqs, True)  # 运行一次prefill
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        # 分配KV缓存并绑定到模型
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()  # 查询当前GPU的空闲和总显存
        used = total - free  # 实际已用显存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 历史分配过的显存峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前分配的显存
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 每个进程分到的KV头数，用于TP并行
        # 计算每个KV缓存块的字节数（2表示k和v，层数*块数*块大小*头数*每头维度*数据类型字节数）
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        # 计算可用的KV缓存块数量（考虑显存利用率、已用、峰值、当前分配等）
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0  # 至少要有一个块
        # 分配KV缓存张量，形状为[2, 层数, 块数, 块大小, 头数, 头维度]
        self.kv_cache = torch.zeros(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, hf_config.head_dim
        )
        layer_id = 0
        # 遍历模型的所有子模块，将分配好的KV缓存绑定到每一层。
        # 在这里会不断遍历具体的算子，实际上成员变量有k_cache和v_cache只有Attention（），所以最终遍历后只有这里的参数会被引用映射到分配好的内存位置
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # 绑定K缓存
                module.v_cache = self.kv_cache[1, layer_id]  # 绑定V缓存
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 构造block_table张量，补齐长度
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 构造 prefill 阶段的输入张量
        input_ids = []         # 存放所有待推理的 token id
        positions = []         # 存放每个 token 的位置信息
        cu_seqlens_q = [0]     # 累加的 query 序列长度（前缀和），用于高效 batch 处理
        cu_seqlens_k = [0]     # 累加的 key 序列长度（前缀和），用于高效 batch 处理
        max_seqlen_q = 0       # 当前 batch 内最大的 query 长度
        max_seqlen_k = 0       # 当前 batch 内最大的 key 长度
        slot_mapping = []      # 存放每个 token 在 KV cache 中的物理位置
        block_tables = None    # block_table 张量，只有 prefix cache 时才用

        for seq in seqs:
            seqlen = len(seq)  # 当前序列的总长度
            # 取出本轮需要推理的 token id（未缓存部分）
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 生成这些 token 的位置信息
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # 计算 query 和 key 的长度
            seqlen_q = seqlen - seq.num_cached_tokens  # 本轮需要推理的 token 数
            seqlen_k = seqlen                          # 当前序列的总长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # 更新 query 前缀和
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # 更新 key 前缀和
            max_seqlen_q = max(seqlen_q, max_seqlen_q)        # 更新最大 query 长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k)        # 更新最大 key 长度
            if not seq.block_table:
                continue  # 如果没有 block_table，跳过后续 slot_mapping 处理
            # 遍历本轮需要写入 KV cache 的 block
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens  # 最后一个 block 可能不满
                slot_mapping.extend(list(range(start, end)))  # 记录每个 token 的物理位置

        # 如果有 prefix cache（key 比 query 多），需要准备 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 转为 CUDA 张量并固定内存，提升推理效率
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置推理上下文，包括 cu_seqlens、slot_mapping、block_tables 等
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions  # 返回模型推理所需的 input_ids 和 positions

    def prepare_decode(self, seqs: list[Sequence]):
        # 构造decode阶段的输入
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 构造采样温度张量
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 执行模型推理（不计算梯度，节省显存和加速）
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 如果是 prefill 阶段，或强制 eager 模式，或 batch size 大于 512
            # 直接用常规方式前向推理并计算 logits
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 否则，使用 CUDA Graphs 加速 decode 阶段
            bs = input_ids.size(0)  # 当前 batch size
            context = get_context()  # 获取当前推理上下文
            # 选择合适 batch size 的 CUDA Graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars  # 获取 CUDA Graph 相关变量
            # 清零所有输入变量（除了 outputs）
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            # 将本次推理的输入数据写入 CUDA Graph 变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 复用 CUDA Graph 进行推理
            graph.replay()
            # 返回本次推理的 logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 统一入口，执行一次推理并采样
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # 捕获不同batch size的CUDA图，加速decode，
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 最大支持的batch size，最多512
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 每个序列最多需要多少个block
        input_ids = torch.zeros(max_bs, dtype=torch.int64)  # 预分配input_ids张量
        positions = torch.zeros(max_bs, dtype=torch.int64)  # 预分配positions张量
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)  # 预分配slot_mapping张量
        context_lens = torch.zeros(max_bs, dtype=torch.int32)  # 预分配context_lens张量
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # 预分配block_tables张量
        outputs = torch.zeros(max_bs, hf_config.hidden_size)  # 预分配输出张量
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # 支持的batch size列表
        self.graphs = {}  # 存放不同batch size的CUDA图
        self.graph_pool = None  # CUDA图池

        # 依次为每种batch size捕获CUDA图（从大到小）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()  # 创建一个CUDA图对象
            # 设置上下文，包括slot_mapping、context_lens、block_tables等
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup，确保相关kernel已编译
            with torch.cuda.graph(graph, self.graph_pool):  # 捕获CUDA图
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture推理过程
            if self.graph_pool is None:
                self.graph_pool = graph.pool()  # 记录第一个图的池
            self.graphs[bs] = graph  # 保存该batch size的CUDA图
            torch.cuda.synchronize()  # 等待所有CUDA操作完成
            reset_context()  # 重置上下文，避免影响后续

        # 保存所有用于CUDA图推理的变量
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
