import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterstats import zonal_stats
import os
import warnings
warnings.filterwarnings("ignore")
pop_tif = r"tif\chn_ppp_2024_1km_Aggregated.tif"  # 人口数据
ntl_tif = r"tif\nightlight.tif"                   # 夜间灯光数据
if not os.path.exists(pop_tif):
    exit()
if not os.path.exists(ntl_tif):
    exit()
df = pd.read_csv("太原市_机器学习特征_含有路网.csv")
geometries = []
for index, row in df.iterrows():
    coords = row['Polygon'].split('|')
    min_lng, min_lat = map(float, coords[0].split(','))
    max_lng, max_lat = map(float, coords[1].split(','))
    geometries.append(box(min_lng, min_lat, max_lng, max_lat))
grids = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
pop_stats = zonal_stats(grids, pop_tif, stats="sum")
grids['人口总数'] = [s['sum'] if s['sum'] else 0 for s in pop_stats]
ntl_stats = zonal_stats(grids, ntl_tif, stats="mean")
grids['夜间灯光亮度'] = [s['mean'] if s['mean'] else 0 for s in ntl_stats]
file_name = "lisafirst/太原市_终极多源特征矩阵.csv"
pd.DataFrame(grids.drop(columns='geometry')).to_csv(file_name, index=False, encoding="utf-8-sig")
