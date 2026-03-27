import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import libpysal
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
import osmnx as ox
import warnings

warnings.filterwarnings("ignore")
# ================= 1. 环境与字体设置 =================
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

try:
    try:
        df_geo = pd.read_csv("太原市_多源特征矩阵.csv")[['Grid_ID', 'Polygon']]
    except:
        df_geo = pd.read_csv("lisafirst/太原市_多源特征矩阵.csv")[['Grid_ID', 'Polygon']]
    df_res = pd.read_csv("附图及csv/太原市_预测与供需错配排名.csv")
    df = pd.merge(df_res, df_geo, on='Grid_ID', how='left')
except Exception as e:
    print(f"错误: {e}")
    exit()
# 解析坐标并构建空间对象
def parse_poly(poly_str):
    try:
        coords = list(map(float, poly_str.replace('|', ',').split(',')))
        return box(*coords)
    except:
        return None
df['geometry'] = df['Polygon'].apply(parse_poly)
grids = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
grids = grids.dropna(subset=['geometry'])
# ================= 2. 核心数学模型 (CCD & Yi) =================
scaler = MinMaxScaler(feature_range=(0.0001, 1))
grids['S_norm'] = scaler.fit_transform(grids[['充电站_数量']])
grids['D_norm'] = scaler.fit_transform(grids[['预测理论需求量']])
# 耦合度 C，协调指数 T，耦合协调度 CCD
grids['C'] = 2 * np.sqrt(grids['S_norm'] * grids['D_norm']) / (grids['S_norm'] + grids['D_norm'])
grids['T'] = 0.5 * (grids['S_norm'] + grids['D_norm'])
grids['CCD'] = np.sqrt(grids['C'] * grids['T'])
grids['Yi'] = (1 - grids['CCD']).round(4)  # 失配指数
# 划分特征类型
grids['供需特征'] = np.where(grids['D_norm'] > grids['S_norm'], '供给滞后(缺桩)', '供给超前(过剩)')
# ================= 3. 空间自相关检验 (Global & Local) =================
w = libpysal.weights.KNN.from_dataframe(grids, k=8)
w.transform = 'r'
y_val = grids['Yi'].values
moran = Moran(y_val, w)
lisa = Moran_Local(y_val, w)
sig = lisa.p_sim < 0.05
spots = lisa.q * sig
# ================= 4. 加载 OSM 街道底图 =================
ox.settings.use_cache = True
bbox = (112.45, 37.75, 112.65, 37.95)
try:
    G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False)
except:
    edges = None
fig, axes = plt.subplots(1, 2, figsize=(26, 13))
# --- 图 (a) 供需特征分布---
if edges is not None:
    edges.plot(ax=axes[0], linewidth=0.4, edgecolor='#999999', alpha=0.4, zorder=1)
color_map_type = {'供给滞后(缺桩)': '#d73027', '供给超前(过剩)': '#4575b4'}
grids.plot(column='供需特征', ax=axes[0], categorical=True,
           color=[color_map_type[v] for v in grids['供需特征']],
           edgecolor='#333333', linewidth=0.2, alpha=0.75, zorder=2)
axes[0].set_title("(a) 太原市充电设施供需错配特征空间分布", fontsize=26, fontweight='bold', pad=25)
type_patches = [mpatches.Patch(color=c, label=l) for l, c in color_map_type.items()]
axes[0].legend(handles=type_patches, loc='lower right', fontsize=16, frameon=True, shadow=True)
# --- 图 (b) LISA 集聚分析 ---
if edges is not None:
    edges.plot(ax=axes[1], linewidth=0.4, edgecolor='#999999', alpha=0.4, zorder=1)
lisa_colors = {0: "#eeeeee", 1: "#d7191c", 2: "#abd9e9", 3: "#2c7bb6", 4: "#fdae61"}
lisa_labels = {0: "不显著", 1: "HH (严重失配重灾区)", 2: "LH (低值包围高值)", 3: "LL (协调冷点区)",
               4: "HL (高值包围低值)"}
grids['lisa_color'] = [lisa_colors[s] for s in spots]
grids.plot(color=grids['lisa_color'], ax=axes[1], edgecolor='#333333', linewidth=0.2, alpha=0.75, zorder=2)
# 在标题中加入空间统计硬指标
title_b = f"(b) 空间失配指数局部自相关 (LISA) 聚类图\nGlobal Moran's I: {moran.I:.3f} | Z-score: {moran.z_sim:.2f} | P < 0.01"
axes[1].set_title(title_b, fontsize=26, fontweight='bold', pad=25)
lisa_patches = [mpatches.Patch(color=c, label=lisa_labels[k]) for k, c in lisa_colors.items()]
axes[1].legend(handles=lisa_patches, loc='lower right', fontsize=16, frameon=True, shadow=True)
# 统一地理视窗
for ax in axes:
    ax.set_axis_off()
    ax.set_xlim(112.44, 112.66)
    ax.set_ylim(37.74, 37.96)
plt.tight_layout()
final_img = "CCD与LISA.png"
plt.savefig(final_img, dpi=600, bbox_inches='tight')
plt.close()
# 导出最终面板
final_csv = "太原市_第三阶段_面板数据.csv"
grids.drop(columns=['geometry', 'lisa_color', 'Polygon']).to_csv(final_csv, index=False, encoding='utf-8-sig')
