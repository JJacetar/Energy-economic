import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ================= 1. 环境与字体设置 =================
# 设置中文字体，防止学术图表里的中文变成方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 如果是Mac系统，请取消这行注释，注释上一行
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 数据加载与预处理 =================
file_name = "太原市_机器学习特征矩阵_最终版.csv"

# 容错机制：如果没有真实数据，生成一点测试数据让你先看效果
if not os.path.exists(file_name):
    print(f"⚠️ 找不到 {file_name}，正在生成模拟数据用于绘图演示...")
    import numpy as np
    lngs, lats = np.meshgrid(np.arange(112.45, 112.65, 0.01), np.arange(37.75, 37.95, 0.01))
    dummy_data = []
    for i in range(lngs.shape[0]-1):
        for j in range(lngs.shape[1]-1):
            polygon_str = f"{lngs[i,j]:.6f},{lats[i,j]:.6f}|{lngs[i+1,j+1]:.6f},{lats[i+1,j+1]:.6f}"
            dummy_data.append({
                "Grid_ID": len(dummy_data)+1, "Polygon": polygon_str,
                "充电站_数量": np.random.randint(0, 5),
                "住宅小区_数量": np.random.randint(0, 50),
                "写字楼_数量": np.random.randint(0, 20)
            })
    df = pd.DataFrame(dummy_data)
else:
    df = pd.read_csv(file_name)
    print("✅ 成功加载真实特征数据！")

# ================= 3. 构造地理空间数据 (GeoDataFrame) =================
# 将 CSV 里的 Polygon 字符串转换为 Shapely 的几何多边形对象
geometries = []
for index, row in df.iterrows():
    # 解析 "min_lng,min_lat|max_lng,max_lat"
    coords = row['Polygon'].split('|')
    min_lng, min_lat = map(float, coords[0].split(','))
    max_lng, max_lat = map(float, coords[1].split(','))
    # 生成矩形几何体
    grid_box = box(min_lng, min_lat, max_lng, max_lat)
    geometries.append(grid_box)

# 转换为专业的地理数据框
gdf = gpd.GeoDataFrame(df, geometry=geometries)

# ================= 4. 绘制顶级学术组图 =================
# 创建一个 2x2 的高清画板 (分辨率 dpi=300 满足核心期刊要求)
fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
fig.suptitle("太原市主城区空间网格划分与核心特征分布热力图", fontsize=20, fontweight='bold', y=0.95)

# ----------------- 子图 1：网格切分示意图 -----------------
ax1 = axes[0, 0]
# 画出网格边框，内部透明
gdf.boundary.plot(ax=ax1, linewidth=0.5, edgecolor='black')
ax1.set_title("(a) 空间网格划分示意 (1km x 1km)", fontsize=14)
ax1.set_axis_off() # 学术地图通常隐藏经纬度坐标轴以保持整洁

# ----------------- 子图 2：供给侧 - 充电站空间分布 -----------------
ax2 = axes[0, 1]
# 使用 Reds (红色渐变) 画热力图
gdf.plot(column='充电站_数量', ax=ax2, cmap='Reds', legend=True,
         legend_kwds={'label': "充电站数量 (个)", 'orientation': "horizontal", 'shrink': 0.6},
         edgecolor='grey', linewidth=0.1)
ax2.set_title("(b) 供给侧：充电站空间分布", fontsize=14)
ax2.set_axis_off()

# ----------------- 子图 3：需求侧 - 住宅小区分布 -----------------
ax3 = axes[1, 0]
# 使用 Blues (蓝色渐变) 画需求特征
gdf.plot(column='住宅小区_数量', ax=ax3, cmap='Blues', legend=True,
         legend_kwds={'label': "住宅小区数量 (个)", 'orientation': "horizontal", 'shrink': 0.6},
         edgecolor='grey', linewidth=0.1)
ax3.set_title("(c) 需求侧：住宅小区空间分布", fontsize=14)
ax3.set_axis_off()

# ----------------- 子图 4：需求侧 - 写字楼/商办分布 -----------------
ax4 = axes[1, 1]
# 使用 Purples (紫色渐变)
gdf.plot(column='写字楼_数量', ax=ax4, cmap='Purples', legend=True,
         legend_kwds={'label': "写字楼数量 (个)", 'orientation': "horizontal", 'shrink': 0.6},
         edgecolor='grey', linewidth=0.1)
ax4.set_title("(d) 需求侧：写字楼/商办空间分布", fontsize=14)
ax4.set_axis_off()

# ================= 5. 调整布局与保存 =================
plt.tight_layout(rect=[0, 0, 1, 0.93]) # 留出大标题的空间
save_path = "太原市_空间特征组图_学术版.png"
plt.savefig(save_path, bbox_inches='tight') # bbox_inches='tight' 切除多余白边
print(f"🎉 顶级学术大图已生成！已保存在当前目录：{save_path}")

plt.show() # 在IDE里展示出来