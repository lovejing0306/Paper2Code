# coding=utf-8
import torch
from torch import sqrt


class DenoiseDiffuion:
	# 核心公式 x_t = sqrt(alpha_t) * x_t-1 + sqrt(1-alpha_t) * noise
	# 准备超参数alpha和beta，代表强度不同的高斯分布
	def __init___(self, num_time_step=1000):
		self.beta = torch.linspace(0.001, 0.02, num_time_step)
		self.alpha = 1 - self.beta
		self.alpha_bar = torch.cumprod(self.alpha) # [alpha_0, alpha_0 * alpha_1, ...]

	# 正向加噪过程 x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
	def q_sample(self, x0, noise, t):
		alpha_bar_t = self.alpha_bar[t]
		x_t =  sqrt(alpha_bar_t) * x0 +  sqrt(1 - alpha_bar_t) * noise
		return x_t
	
	def p_sample(self, model, x_t, t):
		# 推导过程还需要再仔细了解
		pred_noise = model(x_t, t)
		alpha_bar_t = self.alpha_bar[t]
		alpha_t = self.alpha[t]
		coef = (1 - alpha_t) / sqrt(1 - alpha_bar_t)
		mean = 1 / sqrt(alpha_t) * (x_t - coef * pred_noise)
		var = self.beta[t]
		eps = torch.randn(x_t.shape) # 额外随机噪声
		return mean + sqrt(var) * eps
			
	# loss计算过程，model负责预测noise，
	# loss = (noise - model(q_sample(x0, t, noise), t)) ** 2
	def loss(self, model, x0):
		t = torch.randint(0, 1000) # 随机抽样t
		noise = torch.randn_like(x0) # （0,1)高斯噪声
		x_t = self.q_sample(x0, noise, t)
		pred_noise = model(x_t, t)
		return (pred_noise - noise) ** 2
