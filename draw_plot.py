import matplotlib.pyplot as plt
import numpy as np

gat_precision = [0.1142, 0.0439, 0.0491, 0.0661, 0.0539, 0.0476, 0.0549, 0.0540, 0.0560, 0.0506,
                 0.0548, 0.0714, 0.0591, 0.0590, 0.0768, 0.0682, 0.0615, 0.0749, 0.0678, 0.0798]

gat_recall = [0.2343, 0.3203, 0.2758, 0.2966, 0.3215, 0.3327, 0.4021, 0.3577, 0.3725, 0.3832,
              0.4241, 0.3553, 0.3571, 0.3885, 0.3891, 0.3517, 0.4087, 0.2948, 0.3802, 0.3476]

gcn_precision = [0.1847, 0.1751, 0.1854, 0.1617, 0.1881, 0.1782, 0.1757, 0.1758, 0.1821, 0.1827,
                 0.1667, 0.1599, 0.1856, 0.1828, 0.1808, 0.1946, 0.1672, 0.1763, 0.1873, 0.1794]

gcn_recall = [0.1542, 0.2094, 0.1922, 0.2461, 0.2011, 0.2088, 0.2165, 0.2046, 0.2052, 0.2011,
              0.2319, 0.2390, 0.1868, 0.2005, 0.1880, 0.1785, 0.2295, 0.2094, 0.1803, 0.2028]

sage_precision = [0.1682, 0.1933, 0.1754, 0.1827, 0.1992, 0.1866, 0.1807, 0.1930, 0.2277, 0.2013,
                  0.1871, 0.1943, 0.1729, 0.2192, 0.1879, 0.2262, 0.1888, 0.1909, 0.2064, 0.1864]

sage_recall = [0.1477, 0.1501, 0.1892, 0.1874, 0.1679, 0.1851, 0.1886, 0.1904, 0.1483, 0.1815,
               0.2254, 0.2367, 0.2853, 0.2218, 0.2681, 0.2189, 0.2639, 0.2479, 0.2361, 0.2610]

my_precision = [0.1602, 0.2246, 0.1925, 0.1566, 0.1686, 0.1825, 0.1743, 0.2149, 0.1970, 0.1946,
                0.1846, 0.1755, 0.1839, 0.1426, 0.2058, 0.1303, 0.1846, 0.1933, 0.1793, 0.1785]

my_recall = [0.2070, 0.2058, 0.2337, 0.2183, 0.1323, 0.2224, 0.2017, 0.1934, 0.1874, 0.1957,
             0.2325, 0.2123, 0.2200, 0.2100, 0.1862, 0.1922, 0.2064, 0.2112, 0.1898, 0.2307]

my_precision_att = [0.1059, 0.1122, 0.1530, 0.1560, 0.1326, 0.1648, 0.1472, 0.1459, 0.1672, 0.1465,
                    0.1519, 0.1289, 0.1632, 0.1857, 0.1628, 0.1060, 0.1952, 0.1557, 0.0882, 0.1282]

my_recall_att = [0.1566, 0.2266, 0.1311, 0.1868, 0.2224, 0.1791, 0.2349, 0.2088, 0.1916, 0.2734,
                 0.2195, 0.2017, 0.1898, 0.2046, 0.1951, 0.2479, 0.1767, 0.2254, 0.3458, 0.2058]


# 数据准备
models = ['GAT', 'GCN', 'SAGE', 'My Model', 'My Model New']
precision_data = [gat_precision, gcn_precision, sage_precision, my_precision, my_precision_att]
recall_data = [gat_recall, gcn_recall, sage_recall, my_recall, my_recall_att]

# 通用样式配置
plt.style.use('ggplot')
colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', 'Red']
line_styles = ['-', '--', '-.', ':', 'solid']
markers = ['o', 's', 'D', '^', 'o']
x_ticks = np.arange(0, 21, 5)  # 优化X轴刻度

# 独立绘制准确率曲线
plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.plot(precision_data[i],
             color=colors[i],
             linestyle=line_styles[i],
             linewidth=2,
             marker=markers[i],
             markersize=8,
             markevery=3,
             label=f'{model} (Max: {max(precision_data[i]):.3f})')

plt.title('Precision Comparison Across Training Steps', fontsize=14, pad=15)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.xticks(x_ticks)
plt.grid(True, alpha=0.4)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.tight_layout()
plt.savefig('precision_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

# 独立绘制召回率曲线
plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.plot(recall_data[i],
             color=colors[i],
             linestyle=line_styles[i],
             linewidth=2,
             marker=markers[i],
             markersize=8,
             markevery=3,
             label=f'{model} (Max: {max(recall_data[i]):.3f})')

plt.title('Recall Comparison Across Training Steps', fontsize=14, pad=15)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.xticks(x_ticks)
plt.grid(True, alpha=0.4)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.tight_layout()
plt.savefig('recall_comparison.png', dpi=120, bbox_inches='tight')
plt.close()


