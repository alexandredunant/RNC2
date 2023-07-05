import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


p = Path.cwd()
# p = Path("/Users/alexdunant/Library/CloudStorage/GoogleDrive-37stu37@gmail.com/My Drive/Projects/rnc2/data")

# Loading data
c = gpd.read_file(p / "catchments[NNriver].shp")
r = gpd.read_file(p / "river_nodecs[DEM][wTarget].shp")

c.drop(["x", "y", "distance"], axis=1, inplace=True)

# merge point geometry to catchment 
c.rename(columns={'join_rid':'rid'}, inplace=True)
df = pd.merge(c, r, how="left", on='rid')
df.drop(["join_Targe", "geometry_y", "TargetID", "x", "y"], axis=1, inplace=True)
df.rename(columns={'geometry_x': 'geometry'}, inplace=True)

# keep only one river node per catchment
r_c = r[r['rid'].isin(c.rid)]

# export to disk
r_c.to_file(p / "river[one_to_one_catchment].shp")
