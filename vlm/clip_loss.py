import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    CLIP 使用的 InfoNCE 对比损失
    用于图像-文本对的对比学习
    """
    
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: 图像特征 (batch_size, embed_dim)
            text_features: 文本特征 (batch_size, embed_dim)
        
        Returns:
            loss: InfoNCE 对比损失
        """
        # 特征归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵 (batch_size, batch_size)
        # logits[i][j] 表示第 i 个图像和第 j 个文本的相似度
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 创建标签：对角线元素为正样本对
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 图像到文本的损失（每一行是一个图像对所有文本的相似度）
        loss_i2t = F.cross_entropy(logits, labels)
        
        # 文本到图像的损失（每一列是一个文本对所有图像的相似度）
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # 总损失是两个方向损失的平均
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss