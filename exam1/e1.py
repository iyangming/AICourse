import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ==== 设置中文字体（防止中文乱码）====
# 自动使用系统中已有的中文字体（如 SimHei）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==== 数据 ====
years = np.array([
    2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022
])
gdp = np.array([
    27.0092, 31.9244, 34.8517, 41.2119, 48.7940, 53.8580,
    59.2963, 64.3563, 68.8858, 74.6395, 83.2035, 91.9281,
    98.6515, 101, 114, 121
])

# ==== 增速计算 ====
growth_rate = [23.1]  # 2007年给定
growth_rate += list(((gdp[1:] - gdp[:-1]) / gdp[:-1]) * 100)

# ==== 绘图 ====
fig, ax1 = plt.subplots(figsize=(12, 6))

# 灰色柱状图表示GDP
bars = ax1.bar(years, gdp, color='gray', width=0.6)
ax1.set_ylabel("GDP（万亿元）", fontsize=12)
ax1.set_title("2007-2022年GDP及增速统计图", fontsize=16, color='black')

# 每个柱子上显示GDP数值
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 创建第二个Y轴绘制折线图
ax2 = ax1.twinx()
ax2.plot(years, growth_rate, color='blue', linewidth=2.5, marker='o', label='GDP增速（%）')
ax2.set_ylabel("GDP增速（%）", fontsize=12)

# 每个点上显示增速百分比
for i, rate in enumerate(growth_rate):
    ax2.text(years[i], rate + 0.5, f'{rate:.1f}%', ha='center', fontsize=8, color='blue')

# 网格、布局和显示
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()