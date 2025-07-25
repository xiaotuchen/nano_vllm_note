from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    # 定义一个 Block 类，用于表示 KV 缓存中的一个块

    def __init__(self, block_id):
        self.block_id = block_id           # 块的唯一编号
        self.ref_count = 0                 # 当前被多少序列引用（引用计数）
        self.hash = -1                     # 当前块内容的哈希值，-1 表示未设置
        self.token_ids = []                # 当前块存储的 token id 列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash                   # 更新块的哈希值
        self.token_ids = token_ids         # 更新块中存储的 token id 列表

    def reset(self):
        self.ref_count = 1                 # 重置引用计数为 1（新分配时被一个序列引用）
        self.hash = -1                     # 重置哈希值为 -1
        self.token_ids = []                # 清空 token id 列表


class BlockManager:
    """
    BlockManager 负责管理 KV 缓存中的 block 单元，实现 block 的分配、回收、哈希查找和缓存复用。
    支持高效的 KV cache 复用和 LLM 推理中的缓存管理。
    """

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size  # 每个 block 的 token 数
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有 block 对象
        self.hash_to_block_id: dict[int, int] = dict()  # 哈希到 block_id 的映射，用于缓存查找
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲 block 的 id 队列
        self.used_block_ids: set[int] = set()  # 已分配 block 的 id 集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 计算一组 token_ids 的哈希值（可选带前缀），用于缓存查找和复用
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        # 分配指定 block_id 的 block，重置其状态并从空闲队列移除
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        # 回收指定 block_id 的 block，放回空闲队列
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        # 判断当前空闲 block 是否足够分配给 seq
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # 为一个序列分配所需的 block，并支持缓存复用
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 没有命中缓存，需要新分配
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1  # 复用已分配的 block
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 回收一个序列占用的所有 block
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 判断是否可以为序列追加一个 block
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 处理序列追加 token 时的 block 分配和哈希更新逻辑
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
