# coding=utf-8

import torch
import torch.nn as nn

class BatchNorm1d:
    """
    手动实现的 Batch Normalization (1D)
    用于全连接层或1D卷积层
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: 特征维度
            eps: 防止除零的小常数
            momentum: 移动平均的动量
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = torch.ones(num_features)  # 缩放参数
        self.beta = torch.zeros(num_features)   # 偏移参数
        
        # 用于推理的移动平均统计量
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        self.training = True
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, num_features) 或 (batch_size, num_features, length)
        """
        if self.training:
            # 训练模式：使用当前批次的均值和方差
            if x.dim() == 2:
                # (batch_size, num_features)
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
            else:
                # (batch_size, num_features, length)
                mean = x.mean(dim=(0, 2))
                var = x.var(dim=(0, 2), unbiased=False)
            
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式：使用移动平均的统计量
            mean = self.running_mean
            var = self.running_var
        
        # 标准化
        if x.dim() == 2:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            out = self.gamma * x_norm + self.beta
        else:
            x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
            out = self.gamma.view(1, -1, 1) * x_norm + self.beta.view(1, -1, 1)
        
        return out
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True


class BatchNorm2d:
    """
    手动实现的 Batch Normalization (2D)
    用于2D卷积层
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        Args:
            num_features: 通道数
            eps: 防止除零的小常数
            momentum: 移动平均的动量
        """
        self.num_features = num_features  # 序列的维度
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        
        # 用于推理的移动平均统计量
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        self.training = True
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
        """
        if self.training:
            # 训练模式：计算当前批次的均值和方差
            # 沿着 batch, height, width 维度计算
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # 推理模式
            mean = self.running_mean
            var = self.running_var
        
        # 标准化并缩放平移
        # 将 mean, var, gamma, beta 的形状调整为 (1, C, 1, 1)
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = gamma * x_norm + beta
        
        return out
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True


# 使用示例
if __name__ == "__main__":
    # 1D Batch Norm 示例
    print("=== BatchNorm1d 示例 ===")
    bn1d = BatchNorm1d(num_features=4)
    x1d = torch.randn(8, 4)  # (batch_size=8, features=4)
    
    bn1d.train()
    out1d = bn1d.forward(x1d)
    print(f"输入形状: {x1d.shape}")
    print(f"输出形状: {out1d.shape}")
    print(f"训练模式 - 输出均值: {out1d.mean(dim=0)}")
    print(f"训练模式 - 输出方差: {out1d.var(dim=0, unbiased=False)}\n")
    
    # 2D Batch Norm 示例
    print("=== BatchNorm2d 示例 ===")
    bn2d = BatchNorm2d(num_features=3)
    x2d = torch.randn(4, 3, 8, 8)  # (batch_size=4, channels=3, height=8, width=8)
    
    bn2d.train()
    out2d = bn2d.forward(x2d)
    print(f"输入形状: {x2d.shape}")
    print(f"输出形状: {out2d.shape}")
    print(f"训练模式 - 每个通道的均值: {out2d.mean(dim=(0, 2, 3))}")
    print(f"训练模式 - 每个通道的方差: {out2d.var(dim=(0, 2, 3), unbiased=False)}\n")
    
    # 推理模式
    bn2d.eval()
    out2d_eval = bn2d.forward(x2d)
    print("推理模式下的输出完成")
    
    # 与 PyTorch 自带的 BatchNorm 对比
    print("\n=== 与 PyTorch BatchNorm 对比 ===")
    torch_bn = nn.BatchNorm2d(3)
    torch_bn.eval()
    with torch.no_grad():
        torch_out = torch_bn(x2d)
    print(f"PyTorch BatchNorm 输出形状: {torch_out.shape}")