import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        LayerNorm实现
        
        Args:
            normalized_shape: 需要归一化的维度大小
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape  # 序列的维度
        self.eps = eps
        
        # 可学习的缩放参数gamma和偏移参数beta
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # 计算均值和方差
        # keepdim=True保持维度用于广播
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和偏移
        output = self.gamma * x_norm + self.beta
        
        return output


# 测试代码
if __name__ == "__main__":
    # 创建LayerNorm层
    batch_size, seq_len, hidden_size = 2, 3, 4
    layer_norm = LayerNorm(hidden_size)
    
    # 生成随机输入
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    output = layer_norm(x)
    
    print("输入shape:", x.shape)
    print("输出shape:", output.shape)
    print("\n输入:\n", x)
    print("\n输出:\n", output)
    
    # 验证归一化效果：每个样本最后一维的均值应接近0，方差应接近1
    print("\n输出的均值(最后一维):", output.mean(dim=-1))
    print("输出的方差(最后一维):", output.var(dim=-1, unbiased=False))
    
    # 与PyTorch官方实现对比
    official_ln = nn.LayerNorm(hidden_size)
    official_ln.weight.data = layer_norm.gamma.data
    official_ln.bias.data = layer_norm.beta.data
    official_output = official_ln(x)
    
    print("\n与官方实现的差异:", torch.max(torch.abs(output - official_output)).item())