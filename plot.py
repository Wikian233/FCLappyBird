import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
a = pd.read_csv('FCLoutput.csv')
d = pd.read_csv('error.csv')

# 创建一个新的图像和坐标轴
fig, ax = plt.subplots(1, 2)

i = 0
# 对于每一列数据，绘制一条线
for column in a.columns:
    ax[0].plot(a[column], linewidth=1, label=f'FCL output')
    i = i + 1

for column in d.columns:
    ax[1].plot(d[column], linewidth=1, label=f'error')

# 添加图例
ax[0].legend()
ax[0].set_title('FCL output')
ax[0].set_xlabel('score')
ax[0].set_ylabel('decision')

ax[1].legend()
ax[1].set_title('error')
ax[1].set_xlabel('score')
ax[1].set_ylabel('y')



# 显示图像
plt.show()