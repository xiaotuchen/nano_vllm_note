import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


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
            y = mask.unsqueeze(1) * y  # 非本卡负责的 token embedding 置零 mask.unsqueeze(1) adds a new axis at position 1,ie shape [n] to [n,1], mask.unsqueeze(1) * y element wise multiply.
            dist.all_reduce(y)  # 多卡间 embedding 求和，聚合所有卡的结果
        return y  # 返回最终 embedding

class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)  # 继承 VocabParallelEmbedding 初始化
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))  # 本卡分片的 bias
            self.bias.weight_loader = self.weight_loader  # 绑定权重加载函数
        else:
            self.register_parameter("bias", None)  # 不使用 bias

    def forward(self, x: torch.Tensor):
        context = get_context()  # 获取当前推理上下文
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1  # 取每个序列最后一个 token 的索引
            x = x[last_indices].contiguous()  # 只取最后一个 token 的输出
        logits = F.linear(x, self.weight, self.bias)  # 线性变换得到 logits. F.linear() atually does x* self_weight_transposed, so x[n,1024]*self_weight_transposed[1024,151936] ->[n,151936]
        if self.tp_size > 1:
            # 多卡时，将各卡 logits 收集到 rank 0 并拼接
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None # torch.empty_like(a) returns a new tensor with the same size, data type, and device as the input tensor a, but with uninitialized values.
            dist.gather(logits, all_logits, 0) # rank0 receives a list of tensors 
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # join all tensors in all_logits along their last dimension. All tensors must have the same shape except for the last dimension.
        return logits  # 返回最终 logits
