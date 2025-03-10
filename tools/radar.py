import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set style for scientific publication
plt.style.use('seaborn-v0_8-paper')
# mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Data preparation
methods = ['MetaIQA', 'MUSIQ', 'CLIP-IQA', 'LIQE', 'SIND']
categories = [
    'KonIQ→SPAQ(in-the-wild)', 'KonIQ→LIVEW(in-the-wild)', 'KonIQ→CID(laboratory)', 'KonIQ→RBID(blur)', 'KonIQ→LIVE(synthetic)',
    'SPAQ→KonIQ(in-the-wild)', 'SPAQ→LIVEW(in-the-wild)', 'SPAQ→CID(laboratory)', 'SPAQ→RBID(blur)', 'SPAQ→LIVE(synthetic)'
]

# Original data
# SRCC
original_values = np.array([
    [0.864, 0.771, 0.756, 0.767, 0.777, 0.689, 0.749, 0.739, 0.710, 0.642],
    [0.872, 0.788, 0.778, 0.806, 0.893, 0.685, 0.731, 0.775, 0.718, 0.844],
    [0.867, 0.804, 0.830, 0.815, 0.864, 0.732, 0.756, 0.733, 0.765, 0.883],
    [0.847, 0.864, 0.798, 0.844, 0.892, 0.842, 0.843, 0.785, 0.787, 0.894],
    [0.888, 0.876, 0.854, 0.869, 0.916, 0.836, 0.846, 0.791, 0.832, 0.916]
])
# (SRCC + PLCC) / 2
original_values = np.array([
    [0.862, 0.782, 0.760, 0.768, 0.753, 0.715, 0.753, 0.755, 0.716, 0.645],
    [0.869, 0.808, 0.782, 0.810, 0.881, 0.703, 0.738, 0.780, 0.718, 0.837],
    [0.866, 0.817, 0.830, 0.824, 0.865, 0.752, 0.773, 0.730, 0.770, 0.889],
    [0.849, 0.854, 0.819, 0.831, 0.898, 0.847, 0.852, 0.767, 0.793, 0.895],
    [0.888, 0.887, 0.854, 0.877, 0.915, 0.850, 0.852, 0.798, 0.835, 0.911],
])

# Normalize each dimension
min_vals = original_values.min(axis=0)
max_vals = original_values.max(axis=0)
normalized_values = np.zeros_like(original_values)
for i in range(original_values.shape[1]):
    normalized_values[:, i] = 0.5 + 0.45 * (original_values[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i]) # 0.5-0.95

# Angle calculations
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
normalized_values = np.concatenate((normalized_values, normalized_values[:, [0]]), axis=1)
angles = np.concatenate((angles, [angles[0]]))

# Create figure
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='polar')

# Colors for different methods (color blind friendly palette)
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']#[::-1]

# Plot data
for i, method in enumerate(methods):
    # Use solid line for SIND, dashed line for others
    linestyle = '-' if method == 'SIND' else '--'
    line = ax.plot(angles, normalized_values[i], linestyle=linestyle, linewidth=2 if method == 'SIND' else 1.5, 
                  label=method, color=colors[i])
    ax.fill(angles, normalized_values[i], alpha=0.1, color=colors[i])
    
    # Add original value annotations
    for j, value in enumerate(original_values[i]):
        angle = angles[j]
        radius = normalized_values[i][j]
        # Adjust the position of text slightly outside the point
        annotation_radius = radius - 0.05
        ax.text(angle, annotation_radius, f'{value:.3f}', 
                ha='center', va='center', fontsize=10, color=colors[i])

# Set chart properties
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])

# 调整标签位置，将标签向外移动
ax.set_xticklabels(categories, fontsize=14)
# 增加标签到中心的距离
ax.tick_params(pad=10)

# Rotate labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45)

# Set radial limits and remove yticks
ax.set_ylim(0.25, 1.0)
ax.set_yticks([0.35, 0.5, 0.65, 0.8, 0.95])
ax.set_yticklabels([])

# Add grid
ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)

# Add legend with custom styling
legend = ax.legend(loc='lower right', bbox_to_anchor=(1.24, -0.1),
                  frameon=True, fontsize=12, ncol=1)
legend.get_frame().set_alpha(0.8)
legend.get_frame().set_edgecolor('gray')

# Add title
# plt.title('Cross-Dataset Evaluation Performance', 
#           pad=20, fontsize=12, fontweight='bold')

# Adjust layout
# plt.tight_layout()

# Save figure
plt.savefig('paper/radar_chart.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('paper/radar_chart.png', format='png', bbox_inches='tight', dpi=300)
plt.show()
