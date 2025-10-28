# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 生成旋转矩阵
def rope_params(max_seq_len, dim):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度 \theta_i
    scale = torch.arange(0, dim, 2).float()[: dim // 2] / dim
    freqs = 1.0 / (10000.0 ** scale)
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(max_seq_len, device=freqs.device)
    # freqs.shape = [max_seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算 m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y] 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    # freqs.shape = [max_seq_len, dim // 2] 
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# 旋转位置编码计算
def rope_apply(x, freqs):
    # xq.shape = [batch_size, seq_len, n_heads, head_dim]
    # xq_.shape = [batch_size, seq_len, n_heads, head_dim // 2, 2]
    batch_size, seq_len, n_heads, head_dim = x.shape
    x_ = x.float().reshape(batch_size, seq_len, n_heads, -1, 2)  # 最后两位是复数位
    
    # 转为复数域
    # [batch_size, seq_len, n_heads, head_dim // 2, 2] -> [batch_size, seq_len, n_heads, head_dim // 2]
    x_ = torch.view_as_complex(x_)
    
    # freqs_cis 需要扩展维度以匹配 [batch_size, seq_len, n_heads, head_dim // 2]
    # [seq_len, head_dim // 2] -> [1, seq_len, 1, head_dim // 2] 以便广播
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    
    # 应用旋转操作，然后将结果转回实数域
    x_out = torch.view_as_real(x_ * freqs)
    x_out = x_out.flatten(-2) # 将结果转为 [batch_size, seq_len, n_heads, head_dim]
    return x_out.type_as(x)


# 注意力类
class Attention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads  # 每个头的维度
        
        # Q, K, V 线性变换层
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        # 输出的投影层
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # 预计算旋转位置编码
        self.freqs = rope_params(max_seq_len * 2, self.head_dim)  # 这里使用的是每个头的维度
    
    def forward(self, x):
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
        
        # 获取当前序列长度对应的旋转位置编码
        freqs = self.freqs[:seqlen]  
        
        # 在注意力操作之前，应用旋转位置编码
        xq = rope_apply(xq, freqs=freqs)
        xk = rope_apply(xk, freqs=freqs)
        
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


# 测试代码
if __name__ == "__main__":
    # 创建注意力层
    attention = Attention(dim=512, n_heads=8, max_seq_len=2048)
    
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