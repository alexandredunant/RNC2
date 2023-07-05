#!/Users/alexdunant/opt/miniconda3/envs/geo/bin/python

"""
Temporal cascade model volcano -> flood risk to critical infrastructure

Volcano (ash)
    -> Catchment
            -> River (depth)
                        -> River
                                -> infrastructure (water depth)
                                
conda install geopandas numba
"""

#%%
import geopandas as gpd
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import math

from mods_volcanic_activity import volcanic_activity_probability, time_to_next_volcanic_activity, geometry_volcano_catchment, ash_volumes_dipersed
from mods_river_flow import advance_river_flow
from mods_weather import generate_daily_rainfall_amount, wind_direction_and_diffspeed_sampling

#%% load path and data
p = Path.cwd().parent / "data"

#p = Path("C:/Users/nszf25/Downloads/data")
R = gpd.read_file(p / "river[one_to_one_catchment].shp")
C = gpd.read_file(p / "catchment_link_to_river.shp")


# %% variables
# starting nodes is 0 for eastern catchment number 2
# starting nodes is 66 for western catchment number 1

# R.flows = np.zeros(R.shape[0], np.float64)
R["flows"] = np.random.randint(0, 1000, size=len(R))

# volcanic activity coordinates
xv, yv = 1908614,5763652

# probability of rain Sep-Nov, Dec-Feb, Mar-May, Jun-Aug
rain_proba = [0.43, 0.4, 0.35, 0.39]
rain_m = [1.62, 1.56, 1.73, 1.8]
rain_s = [1.01, 1.06, 1.09, 1.07]

# global time - should it be hourly?
TIME = 0

#%%
# volcanic activity
# (local) volcanic process time
t_volc = 0

t_next_volc = time_to_next_volcanic_activity(0.93, 0.19, 0) # should s be 0? when was the last explosion? annual time?
probability_volc = volcanic_activity_probability(0.93, 0.19, t_next_volc) # probability of the next explosion? what if no explosion? does it reset?


# %%
for TIME in range(1000):
        
        # weather
        season = 0
        daily_rain = generate_daily_rainfall_amount(rain_proba[season], rain_m[season], rain_s[season])
        wind_dir, diffusion_speed = wind_direction_and_diffspeed_sampling()
        
        # volcanic activities
        explosion_time = 0
        t_next_volc = time_to_next_volcanic_activity(0.93, 0.19, explosion_time) # in years
        probability_volc = volcanic_activity_probability(0.93, 0.19, t_next_volc)
        t_next_volc = t_next_volc * 365 # in days
        probability_volc = probability_volc / 365 # daily probability
        # volcanic explosion?
        if np.random.uniform(0,1) > probability_volc:
                # explosion!
                dist_volc_catch, bear_volc_catch = geometry_volcano_catchment(xv, yv, C.geometry)
                vol_ash = ash_volumes_dipersed(col_height_alpha, col_height_beta)
                ash_catch = ash_thickness(...)
                
        
        # catchment
        

        # river flows
        R["flows"]=advance_river_flow(R)
        