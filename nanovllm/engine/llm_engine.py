import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


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
        # 使用spawn方式创建多进程上下文 It creates a completely new Python interpreter process. The parent process's Python code is imported fresh in the child process, ensuring a clean fresh start.
        ctx = mp.get_context("spawn")
        # 启动tensor parallel的worker进程（主进程为0号，worker从1开始）
        # tp的多进程管理
        for i in range(1, config.tensor_parallel_size): # only for worker
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

    def exit(self):
        """
        清理资源，关闭所有进程和模型runner。
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join() #等待所有子进程结束


    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加推理请求，将prompt编码为token id并加入调度队列。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

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

    def is_finished(self):
        """
        判断所有序列是否推理完成。
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成接口，支持进度条显示，返回解码后的文本和token id。
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        # 如果sampling_params不是列表，则扩展为与prompts等长的列表，正常来说每条prompt有一套自己的sampling_params
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
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
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 收集输出
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
