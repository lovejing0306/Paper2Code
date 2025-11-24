# coding=utf-8
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, dim, n_head):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0, "dim 必须能被 n_head 整除"
        
        self.dim = dim
        self.n_head = n_head
        self.d_k = dim // n_head  # 每个头的维度
        
        # Q, K, V 的线性变换
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        
        # 输出的线性变换
        self.W_o = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        # 线性变换
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分割成多个头 (split_heads逻辑)
        batch_size, seq_len, dim = Q.size()
        # reshape: (batch_size, seq_len, n_head, d_k)
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_k)
        K = K.view(batch_size, seq_len, self.n_head, self.d_k)
        V = V.view(batch_size, seq_len, self.n_head, self.d_k)
        # transpose: (batch_size, n_head, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 缩放点积注意力 (scaled_dot_product_attention逻辑)
        # Q, K, V shape: (batch_size, n_head, seq_len, d_k)
        
        # 计算注意力分数
        scores = Q@K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        # 应用mask(可选)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        attn_output = attention_weights@V
        
        # 合并多个头 (combine_heads逻辑)
        batch_size, n_head, seq_len, d_k = attn_output.size()
        # transpose: (batch_size, seq_len, n_head, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # reshape: (batch_size, seq_len, dim)
        output = attn_output.view(batch_size, seq_len, self.dim)
        
        # 最后的线性变换
        output = self.W_o(output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络(FFN)"""
    
    def __init__(self, dim, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, dim)
    
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder 层"""
    
    def __init__(self, dim, n_head, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头注意力
        self.multi_head_attention = MultiHeadAttention(dim, n_head)
        
        # 前馈神经网络
        self.feed_forward = FeedForward(dim, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + Layer Norm
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + Layer Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """完整的 Transformer Encoder"""
    
    def __init__(self, num_layers, dim, n_head, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, n_head, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 示例使用
if __name__ == "__main__":
    # 超参数
    batch_size = 2
    seq_len = 10
    dim = 512
    n_head = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1
    
    # 创建模型
    encoder = TransformerEncoder(num_layers, dim, n_head, d_ff, dropout)
    
    # 创建输入数据 (batch_size, seq_len, dim)
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = encoder(x)
    
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")
    print(f"\n模型参数总数: {sum(p.numel() for p in encoder.parameters()):,}")