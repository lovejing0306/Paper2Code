import torch
import torch.nn as nn

class AbsolutePE(nn.Module):
    def __init__(self, sqen_len, dim):
        super().__init__()
        scale = torch.arange(0, dim, 2).float() / dim
        freqs = 1/(10000.0 ** scale)

        pos = torch.arange(sqen_len).float()
        freqs = torch.outer(pos, freqs)

        pe = torch.zeros(sqen_len, dim)
        pe[:, 0::2] = torch.sin(freqs)  # 0::2 表示从 0 开始每间隔一个位置放一个元素
        pe[:, 1::2] = torch.cos(freqs)  # 1::2 表示从 1 开始每间隔一个位置放一个元素
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe


if __name__ == '__main__':
    # 示例用法
    batch_size = 2
    seq_len = 10
    dim = 512
    
    # 创建位置编码层
    pos_embedding = AbsolutePE(seq_len, dim)
    
    # 创建输入张量
    x = torch.randn(batch_size, seq_len, dim)
    
    # 应用位置编码
    output = pos_embedding(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")