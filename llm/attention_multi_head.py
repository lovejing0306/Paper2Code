# coding=utf-8
import math
import torch
import torch.nn as nn  
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""
    def __init__(self, dim, num_heads, dropout=0.1):
        """
        初始化多头注意力层
        Args:
            dim: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 线性变换层，用于生成Q、K、V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        
        # 输出投影层
        self.W_o = nn.Linear(dim, dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enable_causal=False):
        """
        前向传播
        Args:
            x: 输入 [batch_size, seq_len, dim]
            enable_mask: 是否启用掩码
        Returns:
            output: 注意力输出 [batch_size, seq_len, dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # 1. 线性变换生成Q、K、V
        Q = self.W_q(x)  # [batch_size, seq_len, dim]
        K = self.W_k(x)  # [batch_size, seq_len, dim]
        V = self.W_v(x)  # [batch_size, seq_len, dim]
        
        # 2. 重塑为多头形式
        # [batch_size, seq_len, dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # 只有在计算 attention 的时候才切分成多头
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数
        # scores = Q * K^T / sqrt(head_dim)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 4. 应用掩码（如果提供）
        if enable_causal:
            mask = torch.ones(scores.size(-2), scores.size(-1)).triu(1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 5. 应用softmax获得注意力权重
        attention_score = F.softmax(scores, dim=-1)
        attention_score = self.dropout(attention_score)
        
        # 6. 计算加权值
        output = attention_score @ V  # [batch_size, num_heads, seq_len, head_dim]
        
        # 7. 重塑回原始形状
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        # 8. 输出投影
        output = self.W_o(output)
        
        return output


# 示例使用
if __name__ == "__main__":
    # 设置参数
    batch_size = 2
    seq_len = 10
    dim = 512
    num_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)

    # 创建多头注意力层
    multi_attention = MultiHeadAttention(dim, num_heads)
    
    # 前向传播
    multi_output = multi_attention(x, enable_causal=False)
    
    print(f"输入形状: {x.shape}")
    print(f"多头注意力输出形状: {multi_output.shape}")
    print(f"注意力头数: {num_heads}")
    print(f"每个头的维度: {dim // num_heads}")
    