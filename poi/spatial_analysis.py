import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
import libpysal
from esda.moran import Moran
import matplotlib.pyplot as plt
import warnings
import os
# ================= 1. 环境配置 =================
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
# ================= 2. 跨文件数据关联=================
res_path = "太原市_预测与供需错配排名.csv"
geo_paths = ["太原市_多源特征矩阵.csv"]
if not os.path.exists(res_path):
    print(f"找不到结果文件: {res_path}")
    exit()
df_res = pd.read_csv(res_path)
# 寻找带有 Polygon 的原始矩阵
df_geo = None
for p in geo_paths:
    if os.path.exists(p):
        print(f"地理信息源: {p}")
        df_geo = pd.read_csv(p)[['Grid_ID', 'Polygon']]
        break
if df_geo is None:
    print("错误")
    exit()
# 执行合并，找回 Polygon
df = pd.merge(df_res, df_geo, on='Grid_ID', how='left')
# 还原地理空间对象
def parse_poly(poly_str):
    try:
        coords = list(map(float, poly_str.replace('|', ',').split(',')))
        return box(*coords)
    except: return None
df['geometry'] = df['Polygon'].apply(parse_poly)
grids = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
grids = grids.dropna(subset=['geometry'])
# ================= 3. 空间变量校准 =================
if 'Yi' not in grids.columns:
    try:
        df_yi = pd.read_csv("太原市_第三阶段_面板数据.csv")[['Grid_ID', 'Yi']]
        grids = pd.merge(grids, df_yi, on='Grid_ID', how='left')
    except:
        exit()
# ================= 4. 空间统计学双重校验 =================
w = libpysal.weights.KNN.from_dataframe(grids, k=8)
w.transform = 'r'
y_yi = grids['Yi'].values
moran_yi = Moran(y_yi, w)
y_gap = grids['供需错配缺口'].values
moran_gap = Moran(y_gap, w)
# ================= 5. 绘制散点图 =================
y_std = (y_yi - y_yi.mean()) / y_yi.std()
wy_std = libpysal.weights.lag_spatial(w, y_std)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_std, wy_std, color='#34495e', alpha=0.4, s=40, edgecolors='white')
b, a = np.polyfit(y_std, wy_std, 1)
ax.plot(y_std, a + b * y_std, color='red', linewidth=2.5, label=f"Moran's I = {moran_yi.I:.4f}")
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title("太原市充电设施空间失配指数 - 莫兰散点图", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("标准化失配指数 (Yi)", fontsize=14)
ax.set_ylabel("空间滞后 (Spatial Lag)", fontsize=14)
ax.legend(loc='upper left', fontsize=12)
# 象限标注
ax.text(grids['Yi'].std()*1.5, grids['Yi'].std()*1.5, 'HH (严重失配区)', color='red', fontweight='bold')
ax.text(-grids['Yi'].std()*2, -grids['Yi'].std()*2, 'LL (协调平衡区)', color='blue', fontweight='bold')
plt.tight_layout()
plt.savefig("莫兰散点图.png", dpi=600, bbox_inches='tight')
plt.close()
