# coding=utf-8
import torch
import torch.nn.functional as F


class FlowMatching:
    # Flow Matching 核心: 学习从噪声到数据的概率路径
    # 使用确定性的线性插值路径: x_t = t * x_1 + (1-t) * x_0
    # 其中 x_0 是噪声, x_1 是真实数据, t ∈ [0,1]
    
    def __init__(self, sigma_min=0.001):
        self.sigma_min = sigma_min  # 最小噪声标准差,避免数值不稳定
    
    # 欧拉采样:从噪声逐步生成数据
    def euler_sample(self, model, x0, num_steps=100):
        """
        使用欧拉方法从噪声生成样本
        dx/dt = v_theta(x_t, t)
        x_{t+dt} = x_t + v_theta(x_t, t) * dt
        """
        dt = 1.0 / num_steps
        x_t = x0.clone()  # 修复: 初始化 x_t
        
        for i in range(num_steps):
            t = torch.full((x0.shape[0],), i * dt, device=x0.device)
            # 预测当前时刻的速度场
            v_pred = model(x_t, t)
            # 欧拉步进
            x_t = x_t + v_pred * dt   # 反向过程 ！！！
        
        return x_t
    
    # Flow Matching 损失函数
    # loss = ||v_theta(x_t, t) - (x_1 - x_0)||^2
    # 模型学习预测从当前点 x_t 到目标数据 x_1 的速度场
    def loss(self, model, x1):
        """
        Flow Matching 训练损失
        model: 速度场预测模型 v_theta(x_t, t)
        x1: 真实数据样本
        """
        batch_size = x1.shape[0]
        # 随机采样时间 t ∈ [0,1]
        t = torch.rand(batch_size, device=x1.device)
        # 从标准高斯分布采样噪声 x_0 ~ N(0, I)
        x0 = torch.randn_like(x1)  # x0 为纯噪声
        
        # 构造条件流: x_t = t * x_1 + (1-t) * x_0
        # x_0 是源分布(通常是高斯噪声), x_1 是目标分布(真实数据)
        x_t = t.view(-1, 1, 1, 1) * x1 + (1 - t.view(-1, 1, 1, 1)) * x0
        # 计算目标速度场(真实流场)
        target_v = x1 - x0
        # 模型预测速度场
        pred_v = model(x_t, t)
        
        # MSE损失: 预测速度场与真实速度场的差异
        loss = F.mse_loss(pred_v, target_v)
        
        return loss
