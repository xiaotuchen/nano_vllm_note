from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256  # 每个 block 的 token 数量
    counter = count() # 全局自增序列号生成器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)  # 分配唯一序列ID
        self.status = SequenceStatus.WAITING  # 初始状态为等待
        self.token_ids = copy(token_ids)      # 保存输入的 token id 列表
        self.last_token = token_ids[-1]       # 记录最后一个 token
        self.num_tokens = len(self.token_ids) # 当前 token 总数
        self.num_prompt_tokens = len(token_ids) # prompt 部分 token 数
        self.num_cached_tokens = 0            # 已缓存的 token 数
        self.block_table = []                 # 记录分配的 block id
        self.temperature = sampling_params.temperature # 采样温度
        self.max_tokens = sampling_params.max_tokens   # 最大生成 token 数
        self.ignore_eos = sampling_params.ignore_eos   # 是否忽略 EOS

    def __len__(self):
        return self.num_tokens  # 返回当前 token 总数

    def __getitem__(self, key):
        return self.token_ids[key]  # 支持下标访问 token

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED  # 判断序列是否已完成

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens  # 已生成的 completion token 数

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]  # prompt 部分的 token id

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]  # completion 部分的 token id

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size  # 已缓存的 block 数

    @property
    def num_blocks(self):
        # 当前序列需要的 block 总数，向上取整
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个 block 实际包含的 token 数
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 获取第 i 个 block 的 token id 列表
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 追加一个生成的 token
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 用于序列化，返回关键信息
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        # 用于反序列化，恢复对象状态
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
