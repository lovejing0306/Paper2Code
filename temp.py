import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rope_params(seq_len, dim):
    scale = torch.arange(0, dim, 2)[:dim//2].float()
    freqs = 1 / (10000.0 ** scale)

    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, freqs)

    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, freqs):
    b, l, n_head, head_dim = x.shape
    x_ = x.float().reshape(b, l , n_head, -1, 2)
    x_ = torch.view_as_complex(x_)
    
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x_ = torch.view_as_real(x_ * freqs)
    x_ = x_.flatten(-2)
    x_ = x_.type_as(x)
    return x_


class RoPE(nn.Module):
    def __init__(self, dim, n_head, max_seq_len):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.n_head = n_head
        self.head_dim  = dim // n_head

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.freqs = rope_params(max_seq_len, self.head_dim)

    def forward(self, x):
        b, l, d = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(b, l, self.n_head, self.head_dim)
        k = k.view(b, l, self.n_head, self.head_dim)
        v = v.view(b, l, self.n_head, self.head_dim)

        freqs = self.freqs[:l]

        q = rope_apply(q, freqs)
        k = rope_apply(k, freqs)
        # q, k, v 都要执行维度交换
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = F.softmax(score, dim=-1)
        attn = score @ v

        attn = attn.transpose(1,2).contiguous().reshape(b, l, d)
        out = self.w_o(attn)
        return out

        
if __name__ == "__main__":
    # 创建注意力层
    attention = RoPE(dim=512, n_head=8, max_seq_len=2048)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 512)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = attention(x)
    
    print(f"输出形状: {output.shape}")
    print(f"模型参数总数: {sum(p.numel() for p in attention.parameters()):,}")
    
    # 验证输出形状是否正确
    assert output.shape == (batch_size, seq_len, 512), f"输出形状不正确: {output.shape}"