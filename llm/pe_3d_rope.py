# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_params(max_seq_len, dim):
    scale = torch.arange(0, dim, 2)[:dim//2].float() / dim
    freqs = 1./(10000. ** scale)
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 计算结果是个复数向量
    # 假设 freqs = [x, y] 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    # freqs.shape = [max_seq_len, dim // 2] 
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, freqs, grid_sizes):
    """
    grid_sizes: [[f, h, w]]
    """
    b, s, n, c = x.shape
    c = c // 2

    # split freqs
    freqs = freqs.split([c - c // 3 * 2, c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes):   # 对每个样本执行计算
        seq_len = f * h * w
        x_i = x[i, :seq_len].float().reshape(seq_len, n, -1, 2)
        x_i = torch.view_as_complex(x_i)
        
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class Attention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

        self.wo = nn.Linear(dim, dim, bias=False)

        self.freqs = torch.cat([
            rope_params(max_seq_len, self.head_dim - self.head_dim//6*4),   # 这里使用的是头的维度
            rope_params(max_seq_len, self.head_dim//6*2), 
            rope_params(max_seq_len, self.head_dim//6*2)
        ], dim=1)

    def forward(self, x, grid_sizes):
        """
        1. 计算线性变化
        2. 执行分头
        3. 注入 rope （在分头基础上执行）
        4. 头个数维度提前
        5. 计算注意力
        """
        bsz, seqlen, _ = x.shape   # [B, S, D]
        # 线性变换得到 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)   # [B, S, D]
        # 重塑为多头形状: (batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)  # [B, S, N, D]
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        # 在注意力操作之前，应用旋转位置编码
        xq = rope_apply(xq, freqs=self.freqs, grid_sizes=grid_sizes)
        xk = rope_apply(xk, freqs=self.freqs, grid_sizes=grid_sizes)
        
        # 转置为 (batch_size, n_heads, seq_len, head_dim) 以便计算注意力
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        
        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        # 应用 softmax 得到注意力权重
        scores = F.softmax(scores, dim=-1)
        
        # 计算加权后的值: attention_weights @ V
        output = scores @ xv  # (bsz, n_heads, seqlen, head_dim)
        
        # 转置回 (batch_size, seq_len, n_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        # 重塑为 (batch_size, seq_len, dim)
        output = output.view(bsz, seqlen, self.dim)
        
        # 最终的线性变换
        output = self.wo(output)
        return output


if __name__ == '__main__':
    grid_sizes = [[5, 8, 8]]
    attention = Attention(dim=512, n_heads=8, max_seq_len=2048)

    # 创建测试输入
    batch_size = 2
    seq_len = grid_sizes[0][0] * grid_sizes[0][1] * grid_sizes[0][2]
    x = torch.randn(batch_size, seq_len, 512)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = attention(x, grid_sizes=grid_sizes)
    print(f"输出形状: {output.shape}")