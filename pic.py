import matplotlib.pyplot as plt
import numpy as np

# 模拟的节点度数据
wn18rr_v1_degrees = np.random.exponential(scale=1.5, size=2746)
fb237k_v1_degrees = np.random.exponential(scale=2.0, size=2000)
cn_82k_ind_degrees = np.random.exponential(scale=2.5, size=65460)
atomic_ind_degrees = np.random.exponential(scale=3.0, size=256570)

# 设置图形布局
plt.figure(figsize=(12, 8))

# WN18RR-v1
plt.hist(wn18rr_v1_degrees, bins=50, alpha=0.6, color='blue', label='WN18RR-v1', density=True)

# FB237k-v1
plt.hist(fb237k_v1_degrees, bins=50, alpha=0.6, color='green', label='FB237k-v1', density=True)

# CN-82K-Ind
plt.hist(cn_82k_ind_degrees, bins=50, alpha=0.6, color='red', label='CN-82K-Ind', density=True)

# ATOMIC-Ind
plt.hist(atomic_ind_degrees, bins=50, alpha=0.6, color='purple', label='ATOMIC-Ind', density=True)

# 添加图例和标题
plt.legend(loc='upper right')
plt.title('Degree Distribution Histogram of Knowledge Graphs')
plt.xlabel('Degree')
plt.ylabel('Density')

# 显示图形
plt.show()