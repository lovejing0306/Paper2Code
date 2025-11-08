# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """CV 中使用的窗口多头自注意力（类似 Swin 的 Window-MSA）"""

    def __init__(self, dim, window_size=(7, 7), num_heads=8, dropout=0.1):
        """
        初始化窗口注意力层。
        Args:
            dim: 通道维度 C
            window_size: 窗口大小 (Wh, Ww)
            num_heads: 注意力头数
            dropout: dropout 概率
        """
        super().__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 线性变换层：生成 Q, K, V
        self.W_q = nn.Linear(dim, dim, bias=True)
        self.W_k = nn.Linear(dim, dim, bias=True)
        self.W_v = nn.Linear(dim, dim, bias=True)
        # 输出投影
        self.W_o = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # # 相对位置偏置表（注意力稳定与提升空间信息建模）
        # Wh, Ww = self.window_size
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        # )

        # # 预计算相对位置索引 [N, N]，其中 N = Wh*Ww
        # N = Wh * Ww
        # coords_h = torch.arange(Wh)
        # coords_w = torch.arange(Ww)
        # # PyTorch 的 meshgrid 默认是 'ij'，此处使用 'ij' 语义
        # coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, Wh, Ww]
        # coords_flat = torch.flatten(coords, 1)  # [2, N]
        # relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, N, N]
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        # relative_coords[:, :, 0] += Wh - 1
        # relative_coords[:, :, 1] += Ww - 1
        # relative_coords[:, :, 0] *= 2 * Ww - 1
        # relative_position_index = relative_coords.sum(-1)  # [N, N]
        # self.register_buffer("relative_position_index", relative_position_index)

        # # 参数初始化
        # nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def window_partition(self, x, window_size):
        """
        将图像特征按窗口划分。
        Args:
            x: 输入特征 [B, H, W, C]
            window_size: (Wh, Ww)
        Returns:
            windows: [B * num_windows, Wh*Ww, C]
        """
        B, H, W, C = x.shape
        Wh, Ww = window_size
        assert H % Wh == 0 and W % Ww == 0, "H与W必须能被window_size整除"
        x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)  # ！！！！！
        # [B, H//Wh, W//Ww, Wh, Ww, C] -> [B*(H//Wh)*(W//Ww), Wh*Ww, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * (H // Wh) * (W // Ww), Wh * Ww, C)
        return windows

    def window_reverse(self, x, window_size, H, W, B):
        """
        将窗口特征还原为原始空间布局。
        Args:
            windows: [B * num_windows, Wh*Ww, C]
            window_size: (Wh, Ww)
            H, W: 原始特征的高宽
            B: batch size
        Returns:
            x: [B, H, W, C]
        """
        Wh, Ww = window_size
        x = x.view(B, H // Wh, W // Ww, Wh, Ww, -1)
        # [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H, W, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        前向传播（窗口注意力）。
        Args:
            x: 输入特征 [B, H, W, C]
            attn_mask: 注意力掩码（可选），用于移位窗口等场景，形状可为
                       [N, N] 或 [Bn, N, N]，其中 N=Wh*Ww，Bn=B*num_windows。
        Returns:
            y: 输出特征 [B, H, W, C]
        """
        B, H, W, C = x.shape
        assert C == self.dim, "输入通道与dim不匹配"
        Wh, Ww = self.window_size
        assert H % Wh == 0 and W % Ww == 0, "H与W必须能被window_size整除"

        # 1) 窗口划分并展平到序列
        windows = self.window_partition(x, self.window_size)  # [Bn, N, C]
        Bn, N, _ = windows.shape

        # 2) 线性变换生成 Q、K、V 并切分多头
        Q = self.W_q(windows).view(Bn, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(windows).view(Bn, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(windows).view(Bn, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 注意力分数 + 相对位置偏置
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [Bn, h, N, N]
        # bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].view(N, N, self.num_heads)
        # bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1, h, N, N]
        # scores = scores + bias

        # 4) 掩码（可选，用于移位窗口）
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [N, N]
                attn_mask = attn_mask[None, None, :, :]  # [1, 1, N, N]
            elif attn_mask.dim() == 3:  # [Bn, N, N]
                attn_mask = attn_mask[:, None, :, :]  # [Bn, 1, N, N]
            scores = scores + attn_mask

        # 5) Softmax与Dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 6) 加权聚合并恢复形状
        y = attn @ V  # [Bn, h, N, head_dim]
        y = y.transpose(1, 2).contiguous().view(Bn, N, C)  # [Bn, N, C]
        y = self.W_o(y)  # 输出投影
        y = self.window_reverse(y, self.window_size, H, W, B)  # [B, H, W, C]
        return y


# 示例使用
if __name__ == "__main__":
    # 图像特征输入（B, H, W, C）
    B, H, W, C = 2, 8, 8, 64
    num_heads = 8
    window_size = (4, 4)

    x = torch.randn(B, H, W, C)
    attn = WindowAttention(dim=C, window_size=window_size, num_heads=num_heads, dropout=0.1)
    y = attn(x)

    print(f"输入形状: {x.shape}")
    print(f"窗口注意力输出形状: {y.shape}")
    print(f"窗口大小: {window_size}, 注意力头数: {num_heads}")
    