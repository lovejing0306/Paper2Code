import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim, dropout=0.0):
        super().__init__()
        assert dim % n_head == 0, "dim 必须能被 n_head 整除"

        self.dim = dim
        self.n_head = n_head
        self.dim_head = dim // n_head

        self.q_w = nn.Linear(dim, dim)
        self.k_w = nn.Linear(dim, dim)
        self.v_w = nn.Linear(dim, dim)
        self.o_w = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, enable_causal=False):
        b, l, d = inputs.shape  # d == self.dim

        q = self.q_w(inputs)
        k = self.k_w(inputs)
        v = self.v_w(inputs)

        # [b, l, dim] -> [b, n_head, l, dim_head]
        q = q.view(b, l, self.n_head, self.dim_head).transpose(1, 2)
        k = k.view(b, l, self.n_head, self.dim_head).transpose(1, 2)
        v = v.view(b, l, self.n_head, self.dim_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim_head)  # [b, n_head, l, l]

        if enable_causal:
            # 上三角为 True，掩盖未来位置
            mask = torch.ones(l, l, device=inputs.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [b, n_head, l, dim_head]
        out = out.transpose(1, 2).contiguous().view(b, l, self.dim)  # 还原到 [b, l, dim]
        outputs = self.o_w(out)
        return outputs
		

if __name__ == '__main__':
	dets = np.array([
        [0,   0,   100, 100, 0.9],  # A
        [10,  10,  110, 110, 0.8],  # B 与 A 有较大重叠，IoU≈0.68
        [200, 200, 300, 300, 0.7],  # C 远离 A/B
        [210, 210, 290, 290, 0.6],  # D 与 C 重叠，IoU≈0.64
    ], dtype=float)
