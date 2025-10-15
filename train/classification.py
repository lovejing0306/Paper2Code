# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNet(nn.Module):
    def __init__(self, input_channel, num_classes): 
        super().__init__()
        self.linear_0 = nn.Linear(input_channel, input_channel)
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(input_channel, num_classes)
        # 移除softmax，因为CrossEntropyLoss内部已包含
    
    def forward(self, x):
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        return x  # 修复：添加return语句


def main():
    # 创建更多样化的训练数据
    torch.manual_seed(42)  # 设置随机种子以便复现
    
    model = SimpleNet(10, 2)
    loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss内部已包含 softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 提高学习率
    
    num_epochs = 100
    batch_size = 5

    for epoch in range(num_epochs):
        # 每个epoch生成新的随机数据
        input_data = torch.randn(batch_size, 10)
        labels = torch.randint(0, 2, (batch_size,))
        
        # 前向传播
        y = model(input_data)
        loss = loss_fn(y, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # 计算准确率
            with torch.no_grad():
                predictions = torch.argmax(y, dim=1)
                accuracy = (predictions == labels).float().mean()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    

if __name__ == '__main__':
    main()
