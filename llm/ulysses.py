# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.distributed as dist


class UlyssesAttention(nn.Module):
    """
    Ulysses 序列并行注意力机制
    通过在序列维度上分割 Q、K、V,实现高效的分布式注意力计算。
    每个 GPU 处理完整的头维度,但只处理部分序列。
    
    参数:
        dim: 模型维度
        num_heads: 注意力头数
        dropout: Dropout 概率
        world_size: 分布式训练的总进程数
        rank: 当前进程的 rank
    """
    
    def __init__(self, dim, num_heads, dropout, world_size, rank):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.world_size = world_size
        self.rank = rank
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        

    def _all_to_all(self, x, scatter_dim, gather_dim):
        """
        执行 all-to-all 通信,在不同维度上分散和收集
        参数:
            x: 输入张量
            scatter_dim: 分散的维度
            gather_dim: 收集的维度
        返回:
            重新分布的张量
        """
        if self.world_size == 1:
            return x
        
        # 获取输入形状
        shape = list(x.shape)  # b, local_l, n_heads, head_dim
        # 在 scatter_dim 上分割
        assert shape[scatter_dim] % self.world_size == 0
        split_size = shape[scatter_dim] // self.world_size
        # 准备输入和输出张量列表
        input_list = list(torch.split(x, split_size, dim=scatter_dim))
        output_list = [torch.empty_like(input_list[0]) for _ in range(self.world_size)]
        # 执行 all-to-all
        dist.all_to_all(output_list, input_list)
        # 在 gather_dim 上连接
        output = torch.cat(output_list, dim=gather_dim)
        return output
    
    def forward(self, x, attention_mask, return_local):
        """
        前向传播
        参数:
            x: shape [batch, local_seq_len, dim] - 已经在序列维度上分割的输入
               在单GPU模式下，local_seq_len = seq_len
               在多GPU模式下，local_seq_len = seq_len // world_size
            attention_mask: 可选的注意力掩码 [batch, num_heads, seq_len, seq_len]
            return_local: 是否返回本地序列片段（不做 all-gather）
        
        返回:
            如果 return_local=True: [batch, local_seq_len, dim]
            如果 return_local=False: [batch, seq_len, dim] (通过 all-gather 得到完整序列)
        """
        x = torch.chunk(x, self.world_size, dim=1)[self.rank]
        batch_size, local_seq_len, dim = x.shape
        
        # 1. 在本地序列片段上进行线性投影
        q = self.q_proj(x)  # [batch, local_seq_len, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 重塑为多头格式
        q = q.view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, local_seq_len, self.num_heads, self.head_dim)
        # 形状: [batch, local_seq_len, num_heads, head_dim]
        
        # 3. All-to-All: 从"序列分割"切换到"头分割"  每个GPU: 部分序列 × 全部头 → 完整序列 × 部分头
        if self.world_size > 1:
            q = self._all_to_all(q, scatter_dim=2, gather_dim=1)
            k = self._all_to_all(k, scatter_dim=2, gather_dim=1)
            v = self._all_to_all(v, scatter_dim=2, gather_dim=1)
        
        # 形状变化: [batch, local_seq_len, num_heads, head_dim] → [batch, seq_len, local_num_heads, head_dim]
        seq_len = q.shape[1]  # 现在是完整序列长度
        local_num_heads = q.shape[2]  # 每个GPU负责的头数
        
        # 4. 转置以进行注意力计算
        q = q.transpose(1, 2)  # [batch, local_num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. 计算注意力分数
        attn_scores = q@k.transpose(-2, -1)  / math.sqrt(self.head_dim)  # [batch, local_num_heads, seq_len, seq_len]
        
        # 6. 应用注意力掩码(如果有)
        if attention_mask is not None:
            # 只使用当前GPU负责的头对应的掩码
            if self.world_size > 1:
                start_head = self.rank * local_num_heads
                end_head = start_head + local_num_heads
                attention_mask = attention_mask[:, start_head:end_head, :, :]
            attn_scores = attn_scores + attention_mask
        
        # 7. Softmax 和 dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 8. 应用注意力权重
        attn_output = attn_weights@v  # [batch, local_num_heads, seq_len, head_dim]
        # 9. 转置回来
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, local_num_heads, head_dim]
        # 10. All-to-All: 从"头分割"切换回"序列分割"  每个GPU: 完整序列 × 部分头 → 部分序列 × 全部头
        if self.world_size > 1:
            attn_output = self._all_to_all(attn_output, scatter_dim=1, gather_dim=2)
        # 形状变化: [batch, seq_len, local_num_heads, head_dim] → [batch, local_seq_len, num_heads, head_dim]
        
        # 11. 重塑并投影输出
        attn_output = attn_output.contiguous().view(batch_size, local_seq_len, dim)
        output = self.out_proj(attn_output)  # [batch, local_seq_len, dim]
        
        # 12. 如果需要完整序列，进行 all-gather
        if self.world_size > 1 and not return_local:
            output_list = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=1)  # [batch, seq_len, dim]
        
        return output
