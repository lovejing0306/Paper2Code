# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """CV 中使用的窗口多头自注意力（类似 Swin 的 Window-MSA）"""

    def __init__(self, dim, window_size=(7, 7), n_head=8, dropout=0.1):
        """
        初始化窗口注意力层。
        Args:
            dim: 通道维度 C
            window_size: 窗口大小 (Wh, Ww)
            n_head: 注意力头数
            dropout: dropout 概率
        """
        super().__init__()
        assert dim % n_head == 0, "dim必须能被n_head整除"

        self.dim = dim
        self.window_size = window_size
        self.n_head = n_head
        self.head_dim = dim // n_head

        # 线性变换层：生成 Q, K, V
        self.w_q = nn.Linear(dim, dim, bias=True)
        self.w_k = nn.Linear(dim, dim, bias=True)
        self.w_v = nn.Linear(dim, dim, bias=True)
        # 输出投影
        self.w_o = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        # # 相对位置偏置表（注意力稳定与提升空间信息建模）
        # Wh, Ww = self.window_size
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * Wh - 1) * (2 * Ww - 1), n_head)
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

    def w_partition(self, x, window_size):
        """
        将图像特征按窗口划分。
        Args:
            x: 输入特征 [B, H, W, C]
            window_size: (Wh, Ww)
        Returns:
            windows: [B * num_windows, Wh*Ww, C]
        """
        bs, h, w, c = x.shape
        w_h, w_w = window_size
        assert h % w_h == 0 and w % w_w == 0, "H与W必须能被window_size整除"
        x = x.view(bs, h // w_h, w_h, w // w_w, w_w, c)  # ！！！！！
        # [B, H//Wh, W//Ww, Wh, Ww, C] -> [B*(H//Wh)*(W//Ww), Wh*Ww, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bs * (h // w_h) * (w // w_w), w_h * w_w, c)
        return windows

    def w_reverse(self, x, window_size, h, w, bs):
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
        w_h, w_w = window_size
        x = x.view(bs, h // w_h, w // w_w, w_h, w_w, -1)
        # [B, H//Wh, W//Ww, Wh, Ww, C] -> [B, H, W, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bs, h, w, -1)
        return x

    def forward(self, x, attn_mask = None):
        """
        前向传播（窗口注意力）。
        Args:
            x: 输入特征 [B, H, W, C]
            attn_mask: 注意力掩码（可选），用于移位窗口等场景，形状可为
                       [N, N] 或 [Bn, N, N]，其中 N=Wh*Ww，Bn=B*num_windows。
        Returns:
            y: 输出特征 [B, H, W, C]
        """
        bs_, h, w, c = x.shape
        assert c == self.dim, "输入通道与dim不匹配"
        w_h, w_w = self.window_size
        assert h % w_h == 0 and w % w_w == 0, "H与W必须能被window_size整除"

        # 1) 窗口划分并展平到序列
        windows = self.w_partition(x, self.window_size)  # [Bn, N, C]
        bs, n, _ = windows.shape

        # 2) 线性变换生成 Q、K、V 并切分多头
        q = self.w_q(windows).view(bs, n, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_k(windows).view(bs, n, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_v(windows).view(bs, n, self.n_head, self.head_dim).transpose(1, 2)

        # 3) 注意力分数 + 相对位置偏置
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [Bn, h, N, N]
        # bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].view(N, N, self.n_head)
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
        out = attn @ v  # [Bn, h, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(bs, n, c)  # [Bn, N, C]
        out = self.w_o(out)  # 输出投影
        out = self.w_reverse(out, self.window_size, h, w, bs_)  # [B, H, W, C]
        return out


# 示例使用
if __name__ == "__main__":
    # 图像特征输入（B, H, W, C）
    B, H, W, C = 2, 8, 8, 64
    n_head = 8
    window_size = (4, 4)

    x = torch.randn(B, H, W, C)
    attention = WindowAttention(dim=C, window_size=window_size, n_head=n_head, dropout=0.1)
    out = attention(x)

    print(f"输入形状: {x.shape}")
    print(f"窗口注意力输出形状: {out.shape}")
    print(f"窗口大小: {window_size}, 注意力头数: {n_head}")
    