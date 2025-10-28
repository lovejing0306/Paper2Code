# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    单层逻辑回归二分类器
    """
    def __init__(self, n_features=2):
        """
        初始化逻辑回归模型
        参数:
            n_features: 特征数量
        """
        # 使用小的随机数初始化权重
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0
    
    def sigmoid(self, z):
        """
        Sigmoid激活函数
        参数:
            z: 线性组合 z = w^T * x + b
        返回:
            sigmoid(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """
        前向传播
        参数:
            X: 输入特征矩阵 (n_samples, n_features)
        返回:
            A: 预测概率 (n_samples, 1)
        """
        # 线性组合: Z = X * w + b
        Z = np.dot(X, self.weights) + self.bias
        # 激活函数: A = sigmoid(Z)
        A = self.sigmoid(Z)
        return A
    
    def compute_loss(self, A, y):
        """
        计算交叉熵损失
        参数:
            A: 预测概率 (n_samples, 1)
            y: 真实标签 (n_samples, 1)
        返回:
            loss: 交叉熵损失
        """
        m = y.shape[0]
        # 交叉熵损失: L = -1/m * sum(y*log(A) + (1-y)*log(1-A))
        loss = -1/m * np.sum(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8))
        return loss
    
    def backward(self, X, A, y):
        """
        反向传播计算梯度
        参数:
            X: 输入特征矩阵 (n_samples, n_features)
            A: 预测概率 (n_samples, 1)
            y: 真实标签 (n_samples, 1)
        返回:
            dw: 权重梯度
            db: 偏置梯度
        """
        m = y.shape[0]
        # 计算梯度
        # dw = 1/m * X^T * (A - y)
        dw = 1/m * np.dot(X.T, (A - y))
        # db = 1/m * sum(A - y)
        db = 1/m * np.sum(A - y)
        return dw, db
    
    def update_parameters(self, dw, db, lr):
        """
        使用梯度下降更新参数
        参数:
            dw: 权重梯度
            db: 偏置梯度
        """
        self.weights -= lr * dw
        self.bias -= lr * db
    
    def train(self, X, y, step, lr):
        """
        训练逻辑回归模型
        参数:
            X: 训练特征矩阵 (n_samples, n_features)
            y: 训练标签 (n_samples, 1) 或 (n_samples,)
        """
        # 确保y是列向量
        if y.ndim == 1:
            y = y.reshape(-1, 1)
    
        # 训练循环
        for i in range(step):
            # 前向传播
            A = self.forward(X)
            # 计算损失
            loss = self.compute_loss(A, y)
            # 反向传播
            dw, db = self.backward(X, A, y)
            # 更新参数
            self.update_parameters(dw, db, lr)
            
            # 打印训练信息
            if i % 2 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")
    
    def predict_proba(self, X):
        """
        预测概率
        参数:
            X: 特征矩阵 (n_samples, n_features)
        返回:
            概率值 (n_samples, 1)
        """
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        参数:
            X: 特征矩阵 (n_samples, n_features)
            threshold: 分类阈值
        返回:
            预测类别 (n_samples, 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """
        计算准确率
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 真实标签 (n_samples, 1) 或 (n_samples,)
        返回:
            准确率
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


# 示例：使用逻辑回归进行二分类
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    # 生成模拟数据
    # 类别0: 均值为[2, 2]
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    y1 = np.zeros((100, 1))
    
    # 类别1: 均值为[6, 6]
    X2 = np.random.randn(100, 2) + np.array([6, 6])
    y2 = np.ones((100, 1))
    
    # 合并数据
    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    
    # 打乱数据
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    
    # 划分训练集和测试集
    split_idx = int(0.8 * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print("训练数据形状:", X_train.shape)
    print("训练标签形状:", y_train.shape)
    
    # 创建并训练模型
    model = LogisticRegression(n_features=2)
    
    print("\n开始训练...")
    model.train(X_train, y_train, step=100, lr=0.0001)
    
    # 评估模型
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\n训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")


"""
逻辑回归反向传播的数学推导
================================

前向传播过程：
--------------
1. Z = X·W + b                    (线性组合)
2. A = sigmoid(Z) = 1/(1+e^(-Z))  (激活函数)
3. L = -1/m * Σ[y·log(A) + (1-y)·log(1-A)]  (交叉熵损失)

反向传播目标：
--------------
我们需要计算：
- ∂L/∂W (损失对权重的梯度)
- ∂L/∂b (损失对偏置的梯度)

完整的链式求导：
--------------

步骤1: 计算 ∂L/∂A (损失对激活值的梯度)
---------------------------------------
L = -1/m * Σ[y·log(A) + (1-y)·log(1-A)]

∂L/∂A = -1/m * [y/A - (1-y)/(1-A)]
      = -1/m * [y(1-A) - (1-y)A] / [A(1-A)]
      = -1/m * (y - A) / [A(1-A)]


步骤2: 计算 ∂A/∂Z (激活函数的导数)
-----------------------------------
A = sigmoid(Z) = 1/(1+e^(-Z))

sigmoid函数的导数性质：
∂A/∂Z = A(1-A)


步骤3: 计算 ∂L/∂Z (使用链式法则)
---------------------------------
∂L/∂Z = ∂L/∂A · ∂A/∂Z
      = [-1/m * (y-A) / [A(1-A)]] · [A(1-A)]
      = -1/m * (y-A)
      = 1/m * (A-y)

这是一个关键的简化！


步骤4: 计算 ∂L/∂W (损失对权重的梯度)
-------------------------------------
因为 Z = X·W + b

∂L/∂W = ∂L/∂Z · ∂Z/∂W
      = [1/m * (A-y)] · X^T
      = 1/m * X^T·(A-y)


步骤5: 计算 ∂L/∂b (损失对偏置的梯度)
-------------------------------------
∂L/∂b = ∂L/∂Z · ∂Z/∂b
      = [1/m * (A-y)] · 1
      = 1/m * Σ(A-y)


关键发现：
----------
虽然我们计算了 loss 的值，但在反向传播中，我们实际上是通过
链式法则逐层求导，最终得到了一个非常简洁的梯度形式：

    dW = 1/m * X^T·(A-y)
    db = 1/m * Σ(A-y)

这个梯度公式中：
- 不需要显式地使用 loss 的数值
- 只需要 A（预测值）和 y（真实值）的差
- 这是交叉熵损失 + sigmoid激活函数的完美组合！


为什么看起来"跳过"了 loss？
---------------------------
实际上我们并没有跳过 loss，而是通过数学推导，将复杂的链式求导
简化成了最终的简洁形式。loss 的计算公式已经隐含在梯度推导中了。

这就像：
- 你想知道一个函数在某点的斜率（梯度）
- 你不需要先计算函数值，再去求导
- 你可以直接通过求导公式得到斜率
"""