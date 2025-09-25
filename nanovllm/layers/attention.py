import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


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
    '''k和v就是要存储的值，k_cache v_cache就是目的地，context.slot_mapping就是存放的索引，这个值是在prepare_Prefill或者prepare_decode中设置好的。
    '''
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1         # 保证最后一维连续,最后一维的步长是1,indicate how far to move in memory to obtain the next element along that axis
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # 保证 head 维度步长正确
    assert k_cache.stride(1) == D and v_cache.stride(1) == D     # 保证 KV cache 步长正确
    assert slot_mapping.numel() == N                             # slot_mapping 数量等于 token 数
    store_kvcache_kernel[(N,)](                                  
        key, key.stride(0),                                      # 传入 key 张量和步长（GPU本地）
        value, value.stride(0),                                  # 传入 value 张量和步长（GPU本地）
        k_cache, v_cache,                                        # 传入全局 KV cache（GPU全局）
        slot_mapping, D                                          # 传入 slot_mapping 和每个 token 的 KV 向量长度（GPU本地）
    )

'''---for notes only start---
def store_kvcache_simplified(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor):
    
    """store_kvcache（）实际的实现逻辑：
    - key：当前步计算的 Key 张量，形状为 ［N，num_heads, head_dim]
    - value： 当前步计算的 Value 张量，形状为 ［N，num_heads, head_dim]
    - k_cache: Key 缓存，形状为 ［max_blocks，num_heads, head_dim]
    - v_cache: Value 缓存，形状为［max_blocks，num_heads, head_dim]
    - slot_mapping：指示每个token 应该存储在缓存中的哪个位置，形状为[N]
    """
    N, num_heads, head_dim = key.shape

    #展平head 和head_dim 维度
    flat_key =key.view(N,-1) # [N,num_heads* head_dim]
    flat_value = value.view(N, -1) # [N,num_heads* head_dim]

    ＃根据slot_mapping将数据存入缓存
    for i in range(N):
        slot = slot_mapping[il.item()
        k_cache[slot] = flat_key[i]
        v_cache[slot] = flat_value[i]

---for notes only end---
'''

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): # numel() returns total number of elements in the tensor, If tensor is empty, returns 0.
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping) # just warmup_model(), k_cache and v_cache still =torch.tensor([]), so will not store_kvcache() for warmup_model() 
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
