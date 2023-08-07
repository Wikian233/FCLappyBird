import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
a = pd.read_csv('layer1.csv')
b = pd.read_csv('layer2.csv')
c = pd.read_csv('layer3.csv')
d = pd.read_csv('error.csv')

# 创建一个新的图像和坐标轴
fig, ax = plt.subplots(2, 2)

i = 0
# 对于每一列数据，绘制一条线
for column in a.columns:
    ax[0, 0].plot(a[column], linewidth=1, label=f'neuron {i} weight')
    i = i + 1
i = 0
for column in b.columns:
    ax[0, 1].plot(b[column], linewidth=1, label=f'neuron {i} weight')
    i = i + 1
i = 0
for column in c.columns:
    ax[1, 0].plot(c[column], linewidth=1, label=f'neuron {i} weight')
    i = i + 1
for column in d.columns:
    ax[1, 1].plot(d[column], linewidth=1, label=f'error')

# 添加图例
ax[0, 0].legend()
ax[0, 0].set_title('layer0 weights')
ax[0, 0].set_xlabel('epoch')
ax[0, 0].set_ylabel('weight')

ax[0, 1].legend()
ax[0, 1].set_title('layer1 weights')
ax[0, 1].set_xlabel('epoch')
ax[0, 1].set_ylabel('weight')

ax[1, 0].legend()
ax[1, 0].set_title('layer2 weights')
ax[1, 0].set_xlabel('epoch')
ax[1, 0].set_ylabel('weight')

ax[1, 1].legend()
ax[1, 1].set_title('error')
ax[1, 1].set_xlabel('epoch')
ax[1, 1].set_ylabel('error')



# 显示图像
plt.show()