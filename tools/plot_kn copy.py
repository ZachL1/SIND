# 1       2       4       8       16      32      64
# 0.3896	0.38415	0.699	0.7812	0.8003	0.8142	0.8079
# 0.3363	0.56015	0.7395	0.80685	0.81955	0.81545	0.824
# 0.4939	0.6539	0.7969	0.8251	0.8133	0.83765	0.82
# 0.6242	0.75095	0.8183	0.8303	0.81925	0.8036	0.7627

import matplotlib.pyplot as plt
import numpy as np

# 数据定义
data = [
    [0.3896, 0.38415, 0.699, 0.7812, 0.8003, 0.8142, 0.8079],
    [0.3363, 0.56015, 0.7395, 0.80685, 0.81955, 0.81545, 0.824],
    [0.4939, 0.6539, 0.7969, 0.8251, 0.8133, 0.83765, 0.82],
    [0.6242, 0.75095, 0.8183, 0.8303, 0.81925, 0.8036, 0.7627]
]

x = [1, 2, 4, 8, 16, 32, 64]  # number of samples per scene (n)
k_values = [1, 2, 4, 8]  # k values for each line

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(8, 6), dpi=300)

# 设置不同的线型和标记
markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['$k$ = 1', '$k$ = 2', '$k$ = 4', '$k$ = 8']

# 绘制折线
for i, (y, k) in enumerate(zip(data, k_values)):
    # 计算每个点的k*n值
    kn_values = [k * n for n in x]
    
    # 创建掩码来标识k*n在[16, 128]范围内的点
    mask = [(kn >= 32) and (kn <= 64) for kn in kn_values]
    
    # 绘制普通部分（较细的线）
    plt.plot(x, y, marker=markers[i], color=colors[i], label=labels[i],
             linewidth=1.5, markersize=8, markerfacecolor='white',
             markeredgewidth=2, alpha=0.5)
    
    # 绘制突出显示部分（较粗的线）
    for j in range(len(x)-1):
        if mask[j] or mask[j+1]:  # 如果任一端点在范围内
            plt.plot(x[j:j+2], y[j:j+2], color=colors[i],
                    linewidth=3, solid_capstyle='round')
            plt.plot([x[j], x[j]], [y[j], y[j]], marker=markers[i],
                    color=colors[i], markersize=8, markerfacecolor='white',
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
plt.setp(l.get_title(), fontsize=16)

# 设置坐标轴范围
plt.ylim(0.3, 0.9)

# 添加网格线
plt.grid(True, which="both", ls="-", alpha=0.2)

# 保存图片
plt.tight_layout()
plt.savefig('paper/kn_plot.pdf', bbox_inches='tight')
plt.savefig('paper/kn_plot.png', bbox_inches='tight', dpi=300)
