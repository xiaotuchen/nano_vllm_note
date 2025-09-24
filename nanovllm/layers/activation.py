import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) # split x into 2 chunks along the last dimension
        return F.silu(x) * y # this is actually silu(col(gate))*col(up_proj)
