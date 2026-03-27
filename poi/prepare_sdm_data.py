import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")
try:
    grids = pd.read_csv("附图及csv/太原市_第三阶段_最终面板数据.csv")
except Exception as e:
    print(f"错误: {e}")
    exit()

grid_geo = gpd.GeoDataFrame(
    grids,
    geometry=gpd.points_from_xy(grids['Center_Lng'], grids['Center_Lat']),
    crs="EPSG:4326"
).to_crs(epsg=32649)
grid_coords_m = np.column_stack((grid_geo.geometry.x, grid_geo.geometry.y))
# 双重编码安全读取
try:
    house_df = pd.read_csv("山西省_太原市2025.csv", encoding='utf-8')
except UnicodeDecodeError:
    house_df = pd.read_csv("山西省_太原市2025.csv", encoding='gbk')

initial_count = len(house_df)
# 只保留 70 年大红本普通住宅，剔除别墅
house_df = house_df[house_df['产权年限'].astype(str).str.contains('70', na=False)]
house_df = house_df[house_df['物业类型'].astype(str).str.contains('住宅', na=False)]
house_df = house_df[~house_df['物业类型'].astype(str).str.contains('别墅', na=False)]
house_df['均价'] = pd.to_numeric(house_df['均价'], errors='coerce')
house_df['竣工年份'] = house_df['竣工时间'].astype(str).str.extract(r'(\d{4})').astype(float)
house_df['房龄'] = (2026 - house_df['竣工年份']).clip(lower=0)
valid_houses = house_df.dropna(subset=['均价', '房龄', 'wgs84lng', 'wgs84lat']).copy()
# 缩尾处理 (Winsorization)
p01 = valid_houses['均价'].quantile(0.01)
p99 = valid_houses['均价'].quantile(0.99)
valid_houses['均价_缩尾'] = valid_houses['均价'].clip(lower=p01, upper=p99)

house_geo = gpd.GeoDataFrame(
    valid_houses,
    geometry=gpd.points_from_xy(valid_houses['wgs84lng'], valid_houses['wgs84lat']),
    crs="EPSG:4326"
).to_crs(epsg=32649)
house_coords_m = np.column_stack((house_geo.geometry.x, house_geo.geometry.y))
dist_matrix = cdist(grid_coords_m, house_coords_m)
dist_matrix = np.maximum(dist_matrix, 1.0)
weights = 1.0 / (dist_matrix ** 2)
sum_weights = np.sum(weights, axis=1)
grids['X1_二手房价_万元'] = ((np.sum(weights * valid_houses['均价_缩尾'].values, axis=1) / sum_weights) / 10000.0).round(4)
grids['X3_平均房龄_年'] = (np.sum(weights * valid_houses['房龄'].values, axis=1) / sum_weights).round(2)
center_geo = gpd.GeoSeries([Point(112.562, 37.866)], crs="EPSG:4326").to_crs(epsg=32649)
grids['X4_距市中心_km'] = (grid_geo.geometry.distance(center_geo[0]) / 1000.0).round(2)
sdm_panel = grids[['Grid_ID', 'Center_Lng', 'Center_Lat', 'Yi', 'X1_二手房价_万元', 'X3_平均房龄_年', 'X4_距市中心_km']].copy()
sdm_panel.rename(columns={'Yi': 'Y_空间失配度'}, inplace=True)
sdm_panel['X2_人口总量_万人'] = (grids['人口总数'] / 10000.0).round(4)
sdm_panel['X5_交通引流能力'] = grids['路网密度_米']
# 对数化处理，补齐 ln_X3_房龄，规避异方差
sdm_panel['ln_X1_房价'] = np.log1p(sdm_panel['X1_二手房价_万元']).round(4)
sdm_panel['ln_X2_人口'] = np.log1p(sdm_panel['X2_人口总量_万人']).round(4)
sdm_panel['ln_X3_房龄'] = np.log1p(sdm_panel['X3_平均房龄_年']).round(4)
sdm_panel['ln_X4_距离'] = np.log1p(sdm_panel['X4_距市中心_km']).round(4)
sdm_panel['ln_X5_交通'] = np.log1p(sdm_panel['X5_交通引流能力']).round(4)
# 确保没有任何NaN会导致模型崩溃
if sdm_panel.isnull().values.any():
    sdm_panel.fillna(0, inplace=True)
final_output = "太原市_第四阶段_SDM面板.csv"
sdm_panel.to_csv(final_output, index=False, encoding='utf-8-sig')
