# coding=utf-8
import torch
import torch.nn as nn  
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """单头注意力机制实现"""
    def __init__(self, dim, dropout=0.1):
        """
        初始化单头注意力层
        
        Args:
            dim: 模型维度
            dropout: dropout概率
        """
        super().__init__()
        self.dim = dim
        
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
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        Returns:
            output: 注意力输出 [batch_size, seq_len, dim]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        
        # 1. 线性变换生成Q、K、V
        Q = self.W_q(x)  # [batch_size, seq_len, dim]
        K = self.W_k(x)  # [batch_size, seq_len, dim]
        V = self.W_v(x)  # [batch_size, seq_len, dim]
        
        # 2. 计算注意力分数
        # scores = Q * K^T / sqrt(dim)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(dim) # scores shape: [batch_size, seq_len, seq_len]
        
        # 3. 应用掩码（如果提供）
        if enable_causal:
            mask = torch.ones(scores.size(-2), scores.size(-1)).triu(1).bool()   # 上三角矩阵表示方阵斜上方存在元素
            scores = scores.masked_fill(mask, float('-inf'))   # 将上三角矩阵中斜上方元素设置为无穷大
        
        # 4. 应用softmax获得注意力权重
        attention_score = F.softmax(scores, dim=-1)
        attention_score = self.dropout(attention_score)
        
        # 5. 计算加权值
        output = attention_score @ V # output shape: [batch_size, seq_len, dim]
        
        # 6. 输出投影
        output = self.W_o(output)
        
        return output


# 示例使用
if __name__ == "__main__":
    # 设置参数
    batch_size = 2
    seq_len = 10
    dim = 512
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建单头注意力层
    attention = SingleHeadAttention(dim)
    
    # 前向传播
    output = attention(x, enable_causal=True)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    