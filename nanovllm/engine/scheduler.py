from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


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

    def is_finished(self):
        """
        判断所有序列是否都已完成（等待和运行队列均为空）。
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        新增一个序列到等待队列。
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度方法，分为prefill和decode两个阶段。
        返回本轮调度的序列列表和是否为prefill阶段。
        """
        # prefill阶段
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 判断token数和KV块是否足够
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            # 如果有调度的序列，返回并标记为prefill阶段
            return scheduled_seqs, True

        # decode阶段
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 如果KV块不足，抢占其他序列
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 将本轮调度的序列重新放回运行队列
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一个序列，将其状态设为WAITING并回收KV块，重新加入等待队列。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理方法，将生成的token追加到序列，并判断是否终止（eos或达到最大token数）。
        如果终止则回收KV块并移出运行队列。
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)