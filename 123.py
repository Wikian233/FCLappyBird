import torch
import torch.nn as nn

class CustomNeuron(nn.Module):
    def __init__(self, input_size):
        super(CustomNeuron, self).__init__()
        
        # 定义权重和偏置
        self.weight = nn.Parameter(torch.randn(input_size, 1))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        # 线性变换
        z = torch.matmul(x, self.weight) + self.bias
        # ReLU激活函数
        return torch.relu(z)

# 使用示例
neuron = CustomNeuron(input_size=5)
input_data = torch.randn(10, 5)  # 10个样本，每个样本有5个特征
output = neuron(input_data)
print(output)