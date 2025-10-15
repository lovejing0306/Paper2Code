# coding=utf-8
import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    """
    绝对位置编码实现
    使用正弦和余弦函数为序列中的每个位置生成唯一的编码向量
    """
    def __init__(self, seq_len, dim):
        """
        初始化绝对位置编码
        Args:
            seq_len (int): 最大序列长度
            dim (int): 模型维度
        """
        super(AbsolutePositionalEmbedding, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        
        # 创建位置编码矩阵
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # 创建位置
        
        # 计算除数项
        # 1 / 10000^(2i/d_model)
        scale = torch.arange(0, dim, 2).float() / dim
        div_term = 1.0 / (10000.0 ** scale)
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        
        # 添加batch维度并注册为buffer（不参与梯度更新 ！！！）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        # 添加位置编码
        x = x + self.pe
        return x


if __name__ == "__main__":
    # 示例用法
    batch_size = 2
    seq_len = 10
    dim = 512
    
    # 创建位置编码层
    pos_embedding = AbsolutePositionalEmbedding(seq_len, dim)
    
    # 创建输入张量
    x = torch.randn(batch_size, seq_len, dim)
    
    # 应用位置编码
    output = pos_embedding(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    