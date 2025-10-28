# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """使用梯度下降的单层线性回归"""
    def __init__(self, n_features):
        """
        初始化线性回归模型
        
        参数:
            n_features: 特征维度
        """
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def mse_loss(self, y_true, y_pred):
        """
        计算均方误差损失
        参数:
            y_true: 真实值
            y_pred: 预测值
        返回:
            MSE损失值
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def train(self, X, y, step, lr):
        """
        训练模型
        
        参数:
            X: 训练特征，形状 (n_samples, n_features)
            y: 训练标签，形状 (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 梯度下降训练
        for i in range(step):
            # 前向传播：计算预测值
            y_pred = np.dot(X, self.weights) + self.bias
            # 计算损失
            loss = self.mse_loss(y, y_pred)
            # 反向传播：计算梯度
            # dL/dw = -2/n * sum((y - y_pred) * x)
            # dL/db = -2/n * sum(y - y_pred)
            dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)
            
            # 更新参数
            self.weights -= lr * dw
            self.bias -= lr * db
            
            # 打印训练进度
            if (i + 1) % 10 == 0:
                print(f"迭代 {i+1}/{step}, 损失: {loss:.4f}")
    
    def predict(self, X):
        """
        预测
        参数:
            X: 特征，形状 (n_samples, n_features)
        返回:
            预测值
        """
        return np.dot(X, self.weights) + self.bias


# 示例：生成模拟数据并训练
if __name__ == "__main__":
    # 设置随机种子以确保可复现
    np.random.seed(42)
    
    # 生成模拟数据：y = 3x + 2 + noise
    n_samples = 100
    X = np.random.randn(n_samples, 1)  # 单特征
    y_true = 3 * X.squeeze() + 2
    noise = np.random.randn(n_samples) * 0.5
    y = y_true + noise
    
    # 创建并训练模型
    model = LinearRegression(n_features=1)
    print("开始训练...")
    model.train(X, y, step=100, lr=0.001)

    # 输出训练结果
    print(f"训练完成！")
    print(f"学习到的权重: {model.weights[0]:.4f} (真实值: 3.0)")
    print(f"学习到的偏置: {model.bias:.4f} (真实值: 2.0)")
    
    # 预测
    y_pred = model.predict(X)
    