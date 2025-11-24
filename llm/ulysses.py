# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class UlyssesAttention(nn.Module):
    """
    Ulysses 序列并行注意力机制
    通过在序列维度上分割 Q、K、V,实现高效的分布式注意力计算。
    每个 GPU 处理完整的头维度,但只处理部分序列。
    
    参数:
        dim: 模型维度
        n_head: 注意力头数
        dropout: Dropout 概率
        world_size: 分布式训练的总进程数
        rank: 当前进程的 rank
    """
    
    def __init__(self, dim, n_head, dropout, world_size, rank):
        super().__init__()
        assert dim % n_head == 0, "dim 必须能被 n_head 整除"
        
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.world_size = world_size
        self.rank = rank
        
        # Q, K, V 投影层
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def _all_to_all(self, x, split_dim, gather_dim):
        """
        执行 all-to-all 通信,在不同维度上分散和收集
        参数:
            x: 输入张量
            split_dim: 分散的维度
            gather_dim: 收集的维度
        返回:
            重新分布的张量
        """
        if self.world_size == 1:
            return x
        
        # 获取输入形状
        shape = list(x.shape)  # b, local_l, n_heads, head_dim
        # 在 split_dim 上分割
        assert shape[split_dim] % self.world_size == 0
        split_size = shape[split_dim] // self.world_size
        # 准备输入和输出张量列表
        input_list = list(torch.split(x, split_size, dim=split_dim))
        output_list = [torch.empty_like(input_list[0]) for _ in range(self.world_size)]
        # 执行 all-to-all
        dist.all_to_all(output_list, input_list)
        # 在 gather_dim 上连接
        output = torch.cat(output_list, dim=gather_dim)
        return output
    
    def forward(self, x, mask, return_local):
        """
        前向传播
        参数:
            x: shape [batch, local_seq_len, dim] - 已经在序列维度上分割的输入
               在单GPU模式下，local_seq_len = seq_len
               在多GPU模式下，local_seq_len = seq_len // world_size
            mask: 可选的注意力掩码 [batch, n_head, seq_len, seq_len]
            return_local: 是否返回本地序列片段（不做 all-gather）
        
        返回:
            如果 return_local=True: [batch, local_seq_len, dim]
            如果 return_local=False: [batch, seq_len, dim] (通过 all-gather 得到完整序列)
        """
        x = torch.chunk(x, self.world_size, dim=1)[self.rank]
        batch_size, local_seq_len, dim = x.shape
        
        # 1. 在本地序列片段上进行线性投影
        q = self.w_q(x)  # [batch, local_seq_len, dim]
        k = self.w_k(x)
        v = self.w_v(x)
        
        # 2. 重塑为多头格式
        q = q.view(batch_size, local_seq_len, self.n_head, self.head_dim)
        k = k.view(batch_size, local_seq_len, self.n_head, self.head_dim)
        v = v.view(batch_size, local_seq_len, self.n_head, self.head_dim)
        # 形状: [batch, local_seq_len, n_head, head_dim]
        
        # 3. All-to-All: 从"序列分割"切换到"头分割"  每个GPU: 部分序列 × 全部头 → 完整序列 × 部分头
        if self.world_size > 1:
            q = self._all_to_all(q, split_dim=2, gather_dim=1)
            k = self._all_to_all(k, split_dim=2, gather_dim=1)
            v = self._all_to_all(v, split_dim=2, gather_dim=1)
        
        # 形状变化: [batch, local_seq_len, n_head, head_dim] → [batch, seq_len, local_n_head, head_dim]
        seq_len = q.shape[1]  # 现在是完整序列长度
        local_n_head = q.shape[2]  # 每个GPU负责的头数
        
        # 4. 转置以进行注意力计算
        q = q.transpose(1, 2)  # [batch, local_n_head, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. 计算注意力分数
        scores = q@k.transpose(-2, -1)  / math.sqrt(self.head_dim)  # [batch, local_n_head, seq_len, seq_len]
        
        # 6. 应用注意力掩码(如果有)
        if mask is not None:
            # 只使用当前GPU负责的头对应的掩码
            if self.world_size > 1:
                start_head = self.rank * local_n_head
                end_head = start_head + local_n_head
                mask = mask[:, start_head:end_head, :, :]
            scores = scores + mask
        
        # 7. Softmax 和 dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # 8. 应用注意力权重
        out = attn@v  # [batch, local_n_head, seq_len, head_dim]
        # 9. 转置回来
        out = out.transpose(1, 2)  # [batch, seq_len, local_n_head, head_dim]
        # 10. All-to-All: 从"头分割"切换回"序列分割"  每个GPU: 完整序列 × 部分头 → 部分序列 × 全部头
        if self.world_size > 1:
            out = self._all_to_all(out, split_dim=1, gather_dim=2)
        # 形状变化: [batch, seq_len, local_n_head, head_dim] → [batch, local_seq_len, n_head, head_dim]
        
        # 11. 重塑并投影输出
        out = out.contiguous().view(batch_size, local_seq_len, dim)
        out = self.w_o(out)  # [batch, local_seq_len, dim]
        
        # 12. 如果需要完整序列，进行 all-gather
        if self.world_size > 1 and not return_local:
            outs = [torch.zeros_like(out) for _ in range(self.world_size)]
            dist.all_gather(outs, out)
            out = torch.cat(outs, dim=1)  # [batch, seq_len, dim]
        
        return out

if __name__ == '__main__':
    torch.manual_seed(0)

    batch = 2
    seq_len = 8
    dim = 16
    n_head = 4

    # world_size=1 避免分布式依赖，测试基本行为
    attn = UlyssesAttention(dim=dim, n_head=n_head, dropout=0.0, world_size=1, rank=0)

    x = torch.randn(batch, seq_len, dim)

    # 构造一个因果掩码：上三角置为大负数，抑制当前位置看到未来
    causal = torch.ones(seq_len, seq_len).triu(1)
    mask = (causal * -1e9).unsqueeze(0).unsqueeze(0).repeat(batch, n_head, 1, 1)

    # 形状与 return_local 行为
    out_full = attn(x, mask=mask, return_local=False)
    out_local = attn(x, mask=mask, return_local=True)
    print(out_full.shape)

    assert out_full.shape == (batch, seq_len, dim)
    assert out_local.shape == (batch, seq_len, dim)

    # world_size=1 时，两种返回方式应一致
    assert torch.allclose(out_full, out_local, atol=1e-6)

    # 掩码应影响输出：与无掩码的结果不同
    out_no_mask = attn(x, mask=None, return_local=False)
    diff = torch.max(torch.abs(out_full - out_no_mask)).item()
    assert diff > 1e-6
