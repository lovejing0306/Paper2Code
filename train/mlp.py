# coding=utf-8

import numpy as np


class MLP:
    """
    多层感知器（Multi-Layer Perceptron）
    使用纯 numpy 实现，支持任意层数和神经元数量
    """
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        """
        初始化 MLP
        参数:
            layer_sizes: list，每层的神经元数量，例如 [784, 128, 64, 10]
            activation: str，激活函数类型 ('relu', 'sigmoid', 'tanh')
            learning_rate: float，学习率
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # He 初始化（适用于 ReLU）或 Xavier 初始化
        for i in range(self.num_layers - 1):
            if activation == 'relu':
                # He 初始化
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier 初始化
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            
            bias = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def relu(self, z):
        """ReLU 激活函数"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU 导数"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip 防止溢出
    
    def sigmoid_derivative(self, z):
        """Sigmoid 导数"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def tanh(self, z):
        """Tanh 激活函数"""
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        """Tanh 导数"""
        return 1 - np.tanh(z) ** 2
    
    def activation(self, z):
        """根据设置选择激活函数"""
        if self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def activation_derivative(self, z):
        """根据设置选择激活函数导数"""
        if self.activation_name == 'relu':
            return self.relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_name == 'tanh':
            return self.tanh_derivative(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def softmax(self, z):
        """Softmax 函数（用于输出层）"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定性
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据，shape = (batch_size, input_size)
        
        返回:
            output: 网络输出
            cache: 缓存的中间值，用于反向传播
        """
        cache = {'A': [X], 'Z': []}
        A = X
        
        # 隐藏层
        for i in range(self.num_layers - 2):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.activation(Z)
            cache['Z'].append(Z)
            cache['A'].append(A)
        
        # 输出层（使用 softmax）
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.softmax(Z)
        cache['Z'].append(Z)
        cache['A'].append(A)
        
        return A, cache
    
    def backward(self, y_true, cache):
        """
        反向传播
        
        参数:
            y_true: 真实标签，shape = (batch_size, num_classes)
            cache: 前向传播缓存的值
        
        返回:
            gradients: 权重和偏置的梯度
        """
        m = y_true.shape[0]  # batch size
        gradients = {'dW': [], 'db': []}
        
        # 输出层梯度（交叉熵损失 + softmax）
        dZ = cache['A'][-1] - y_true
        
        # 反向传播
        for i in range(self.num_layers - 2, -1, -1):
            # 计算权重和偏置的梯度
            dW = np.dot(cache['A'][i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            gradients['dW'].insert(0, dW)
            gradients['db'].insert(0, db)
            
            # 如果不是第一层，继续传播梯度
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.activation_derivative(cache['Z'][i-1])
        
        return gradients
    
    def update_parameters(self, gradients):
        """
        更新权重和偏置
        
        参数:
            gradients: 计算得到的梯度
        """
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * gradients['dW'][i]
            self.biases[i] -= self.learning_rate * gradients['db'][i]
    
    def compute_loss(self, y_true, y_pred):
        """
        计算交叉熵损失
        
        参数:
            y_true: 真实标签（one-hot 编码）
            y_pred: 预测概率
        
        返回:
            loss: 交叉熵损失
        """
        m = y_true.shape[0]
        # 添加小的 epsilon 防止 log(0)
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=True):
        """
        训练 MLP
        
        参数:
            X_train: 训练数据
            y_train: 训练标签（one-hot 编码）
            epochs: 训练轮数
            batch_size: 批量大小
            verbose: 是否打印训练信息
        """
        n_samples = X_train.shape[0]
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_correct = 0
            
            # 批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred, cache = self.forward(X_batch)
                
                # 计算损失
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss * X_batch.shape[0]
                
                # 计算准确率
                predictions = np.argmax(y_pred, axis=1)
                labels = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == labels)
                
                # 反向传播
                gradients = self.backward(y_batch, cache)
                
                # 更新参数
                self.update_parameters(gradients)
            
            # 记录历史
            avg_loss = epoch_loss / n_samples
            accuracy = epoch_correct / n_samples
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            # 打印信息
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return history
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入数据
        
        返回:
            predictions: 预测的类别
            probabilities: 预测的概率
        """
        probabilities, _ = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
            X_test: 测试数据
            y_test: 测试标签（one-hot 编码）
        
        返回:
            accuracy: 准确率
            loss: 损失
        """
        y_pred, _ = self.forward(X_test)
        loss = self.compute_loss(y_test, y_pred)
        
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == labels)
        
        return accuracy, loss


if __name__ == "__main__":
    # 示例：使用 MNIST 风格的数据
    print("=== MLP 示例 ===\n")
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    # 创建简单的分类数据
    X = np.random.randn(n_samples, n_features)
    # 创建 one-hot 标签
    y_labels = np.random.randint(0, n_classes, n_samples)
    y = np.eye(n_classes)[y_labels]
    
    # 分割数据
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 创建 MLP
    mlp = MLP(
        layer_sizes=[n_features, 64, 32, n_classes],
        activation='relu',
        learning_rate=0.01
    )
    
    print(f"网络结构: {mlp.layer_sizes}")
    print(f"激活函数: {mlp.activation_name}")
    print(f"学习率: {mlp.learning_rate}\n")
    
    # 训练
    print("开始训练...\n")
    history = mlp.train(X_train, y_train, epochs=50, batch_size=32, verbose=True)
    
    # 评估
    print("\n评估模型...")
    train_acc, train_loss = mlp.evaluate(X_train, y_train)
    test_acc, test_loss = mlp.evaluate(X_test, y_test)
    
    print(f"\n训练集 - 准确率: {train_acc:.4f}, 损失: {train_loss:.4f}")
    print(f"测试集 - 准确率: {test_acc:.4f}, 损失: {test_loss:.4f}")
    
    # 预测示例
    print("\n预测示例（前5个测试样本）:")
    predictions, probabilities = mlp.predict(X_test[:5])
    true_labels = np.argmax(y_test[:5], axis=1)
    
    for i in range(5):
        print(f"样本 {i+1}: 预测={predictions[i]}, 真实={true_labels[i]}, "
              f"概率={probabilities[i]}")