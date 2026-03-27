import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox
import warnings
import urllib3
import time

# 关闭烦人的 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

df = pd.read_csv("太原市_机器学习特征.csv")

geometries = []
for index, row in df.iterrows():
    coords = row['Polygon'].split('|')
    min_lng, min_lat = map(float, coords[0].split(','))
    max_lng, max_lat = map(float, coords[1].split(','))
    geometries.append(box(min_lng, min_lat, max_lng, max_lat))

grids = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

ox.settings.timeout = 500
ox.settings.requests_kwargs = {'verify': False}
ox.settings.log_console = True  #
ox.settings.use_cache = True  #

north, south = 37.95, 37.75
east, west = 112.65, 112.45
bbox = (west, south, east, north)

G = None
success = False
attempt = 1

while not success:
    try:
        G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
        gdf_nodes, edges = ox.graph_to_gdfs(G)
        success = True

    except Exception as e:
        time.sleep(60)
        attempt += 1

ox.settings.log_console = False

# 投影到 UTM 49N 计算真实米数
grids_proj = grids.to_crs(epsg=32649)
edges_proj = edges.to_crs(epsg=32649)

intersected = gpd.overlay(edges_proj, grids_proj, how='intersection')
intersected['road_length_m'] = intersected.geometry.length
road_lengths = intersected.groupby('Grid_ID')['road_length_m'].sum().reset_index()

final_df = df.merge(road_lengths, on='Grid_ID', how='left')
final_df['路网密度_米'] = final_df['road_length_m'].fillna(0)
final_df.drop(columns=['road_length_m'], inplace=True)

feature_cols = [col for col in final_df.columns if "数量" in col or "路网" in col]
final_df = final_df.loc[(final_df[feature_cols] != 0).any(axis=1)]

file_name = "太原市_机器学习特征_含有路网.csv"
final_df.to_csv(file_name, index=False, encoding="utf-8-sig")
