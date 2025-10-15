# coding=utf-8
import torch
from torch import sqrt


## 先记录下，后序再优化

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

class DenoiseDiffuion:
	# 核心公式 x_t = sqrt(alpha_t) * x_t-1 + sqrt(1-alpha_t) * noise
	# 准备超参数alpha和beta，代表强度不同的高斯分布
	def __init___(self, num_time_step=1000):
			self.beta = torch.linspace(0.001, 0.02, num_time_step)
			self.alpha = 1 - self.beta
			self.alpha_bar = torch.cumprod(self.alpha) # [alpha_0, alpha_0 * alpha_1, ...]

	# 正向加噪过程 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
	def q_sample(self, x0, t, noise):
			alpha_bar_t = self.alpha_bar[t]
			x_t =  sqrt(alpha_bar_t) * x0 +  sqrt(1 - alpha_bar_t) * noise
			return x_t
	
	def p_sample(self, model, x_t, t):
			# 推导过程还需要再仔细了解
			noise = model(x_t, t)
			alpha_bar_t = self.alpha_bar[t]
			alpha_t = self.alpha[t]
			coef = (1 - alpha_t) / sqrt(1 - alpha_bar_t)
			mean = 1 / sqrt(alpha_t) * (x_t - coef * noise)
			var = self.beta[t]
			eps = torch.randn(x_t.shape) # 额外随机噪声
			return mean + sqrt(var) * eps
			
	# loss计算过程，model负责预测noise，
	# loss = (noise - model(q_sample(x0, t, noise), t)) ** 2
	def loss(self, model, x0):
			t = torch.randint(0, 1000) # 随机抽样t
			noise = torch.randn_like(x0) # （0,1)高斯噪声
			x_t = self.q_sample(x0, t, noise)
			pred_noise = model(x_t, t)
			return (pred_noise - noise) ** 2