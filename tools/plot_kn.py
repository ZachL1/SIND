# 1       2       4       8       16      32      64
# 0.3896	0.38415	0.699	0.7812	0.8003	0.8142	0.8079
# 0.3363	0.56015	0.7395	0.80685	0.81955	0.81545	0.824
# 0.4939	0.6539	0.7969	0.8251	0.8133	0.83765	0.82
# 0.6242	0.75095	0.8183	0.8303	0.81925	0.8036	0.7627

import matplotlib.pyplot as plt
import numpy as np

data = [
    [0.3896, 0.38415, 0.699, 0.7812, 0.8003, 0.8142, 0.8079],
    [0.3363, 0.56015, 0.7395, 0.80685, 0.81955, 0.81545, 0.824],
    [0.4939, 0.6539, 0.7969, 0.8251, 0.8133, 0.83765, 0.82],
    [0.6242, 0.75095, 0.8183, 0.8303, 0.81925, 0.8036, 0.7627]
]

x = [1, 2, 4, 8, 16, 32, 64]  # number of samples per scene (n)

# 设置图表样式
# plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(8, 5.5), dpi=300)

# 设置不同的线型和标记
markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['$k$ = 1', '$k$ = 2', '$k$ = 4', '$k$ = 8']  # number of scenes

# 绘制折线
for i, y in enumerate(data):
    plt.plot(x, y, marker=markers[i], color=colors[i], label=labels[i],
             linewidth=2, markersize=8, markerfacecolor='white',
             markeredgewidth=2)

# 设置坐标轴
plt.xscale('log', base=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Number of Samples per Scene ($n$)', fontsize=18)
plt.ylabel('Performance', fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 设置图例
l = plt.legend(title='Number of Scenes', loc='lower right', frameon=True, fontsize=14, ncol=2)
# 设置图例标题的字体大小
plt.setp(l.get_title(), fontsize=16)

# 设置坐标轴范围
plt.ylim(0.3, 0.9)

# 添加网格线
plt.grid(True, which="both", ls="--", alpha=0.7)

# 保存图片
plt.tight_layout()
plt.savefig('paper/kn_plot.pdf', bbox_inches='tight')
plt.savefig('paper/kn_plot.png', bbox_inches='tight', dpi=300)










# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # 数据
# data = {
#     '2': [0.7976, np.nan, np.nan, np.nan],
#     '4': [0.8087, 0.8187, np.nan, np.nan],
#     '8': [0.8243, 0.8307, 0.8239, np.nan],
#     '16': [0.8078, 0.8327, 0.8244, 0.8336],
#     '32': [np.nan, 0.8197, 0.8319, 0.8356],
#     '64': [np.nan, np.nan, 0.8316, 0.8272],
#     '|D|': [0.7438, 0.7756, 0.786, 0.78],
# }

# df = pd.DataFrame(data, index=[16, 32, 64, 128])

# # x轴标签
# x_labels = ['2', '4', '8', '16', '32', '64']
# x_label_special = '|D|'

# # 绘制通常情况的折线
# for i, (index, row) in enumerate(df.iterrows()):
#     # 使用 plt.plot 返回的 Line2D 对象获取颜色
#     line, = plt.plot(x_labels, row[:-1], marker='o', label=f'Batch Size {index}')  # 绘制前六个点
#     color = line.get_color()  # 获取绘制的颜色
#     plt.plot(x_label_special, row[-1], marker='o', color=color)  # 使用相同颜色单独绘制'|D|'点

#     # 获取最后一个有效点的坐标
#     last_valid_index = row[:-1].last_valid_index()
#     last_valid_x = last_valid_index
#     last_valid_y = row[last_valid_index]
    
#     # 绘制从最后一个有效点到 '|D|' 点的线段
#     plt.plot([last_valid_x, x_label_special], [last_valid_y, row[-1]], linestyle='--', color=color)

# # 添加x轴分割线
# plt.axvline(x=5.5, color='grey', linestyle='--', alpha=0.7)  # 在'64'和'|D|'之间画一条竖线，x=5.5是'64'和'|D|'之间的位置

# # 设置x轴
# plt.xticks(ticks=range(len(x_labels) + 1), labels=x_labels + [x_label_special])
# plt.xticks(rotation=45)

# # 设置y轴标签和范围
# plt.ylabel('Value', fontsize=12)
# plt.ylim(0.74, 0.86)

# # 添加图例
# plt.legend(title='Batch Size', loc='upper left')

# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.7)

# # 调整布局并显示图表
# plt.tight_layout()
# plt.savefig('paper/plot_bs_bd.png')
# plt.show()
