# %%
import pandas as pd
import geopandas as gpd
import numpy as np
import os

from mods_catchment_flow import *
from mod_river_flow import *

os.chdir("/Users/alexdunant/Library/CloudStorage/OneDrive-DurhamUniversity/workfolder/rnc2_project/code")

# Define catchments and their corresponding parameters
# catchments = pd.read_csv("../docs/Catchments_updt_sorted.csv", index_col=0)
catchments = gpd.read_file("../docs/Catchments_river_nodes.gpkg")
catchments = catchments[catchments['River System'] == 'Rangitaki']
catchments = catchments.sort_values('Source_river')
catchments['streamflow'] = np.zeros(len(catchments))
catchments['storage'] = np.zeros(len(catchments))
catchments['max_soil_moisture'] = np.random.randint(200, 400, len(catchments))

# Convert df into a dictionary
river_network = {}
water_flow = {}
keys = catchments.Source_river
target_nodes = catchments.Target_river
baseline_river = 50  # baseline value for river flow
catchment_names = catchments['Catchment Name']

# create river network where each edge ßis a tuple of the form (target_node, flow_rate) and a water flow dict (start
# with a mean value of 0 m3/s)
for i in range(len(keys)):
    river_network[keys[i]] = {'target_river': target_nodes[i], 
                              'edge_flow': baseline_river,
                              'catchment_name': catchment_names[i]}
    water_flow[keys[i]] = baseline_river

# generate a list of precipitation for the time of simulation (in days)
num_days = 1000
precipitations = generate_daily_rainfall_amount(num_days)

record = []
df_catchments = catchments.copy()


#%% run algo                          
# Iterate over timesteps
for t in trange(1, num_days):
    # generate daily rainfall amount
    precipitation = precipitations[t]

    # NLRRM model for the catchment - storage and streamflow updated for each catchment
    df_catchments = simulate_catchment_reservoir_flow(df_catchments, precipitation, dtime=1)

    # Create the total flow being routed from catchments to each river node per day
    water_flow = catchment_flow_to_river_nodes(df_catchments, water_flow=water_flow, baseflow=baseline_river)

    # simulate river behavior for a number of stepped iteration (hours?) per day, the last parameter 
    # represent a loss function to prevent flows from stacking up
    water_flow = simulate_river_flow(river_network, water_flow, 18, 0.95)

    # update the new river flow state at the end of the day
    for node in river_network:
        if water_flow[node] == 0:
            river_network[node]['edge_flow'] == baseline_river
        else:
            river_network[node]['edge_flow'] = water_flow[node]

    #  Record results
    df = pd.concat([pd.Series(river_network.keys()),
                    pd.Series([river_network[key]['target_river'] for key in river_network]),
                    pd.Series([river_network[key]['edge_flow'] for key in river_network])], axis=1)
    df.columns = ['source', 'target', 'river_flow']
    df['precipitation'] = precipitation
    df['storage'] = df_catchments['storage']
    df['time'] = t

    record.append(df)

    results = pd.concat(record)

print("simulation finished")


res = results[["time", "precipitation", "storage", "river_flow"]].groupby('time').max().reset_index()





#%%
################################################################
# Plot results
################################################################

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

# create the figure and the axes
# create the figure
fig = plt.figure(figsize=(12,3))

ax1 = fig.add_subplot(1, 1, 1)

# set the x-axis and the first y-axis (precipitation)
ax1.set_xlabel('Time')
ax1.set_ylabel('Precipitation', color='blue')
ax1.step(res.time, res.precipitation, lw=0.5, color='blue')

# create the second y-axis (discharge)
ax2 = ax1.twinx()
ax2.set_ylabel('Discharge', color='red')
ax2.plot(res.time, res.river_flow, linestyle='--', lw=0.5, color='red')

# set the y-axis color to green
ax1.tick_params(axis='y', colors='blue')
ax2.tick_params(axis='y', colors='red')

# set the title of the plot
plt.title('Precipitation and Discharge vs Time')

# display the plot
plt.tight_layout()
plt.show()


# %%
fig = plt.figure(figsize=(5,5))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.hist(results.river_flow, edgecolor="red", facecolor="none", bins=20, label="discharge")
ax1.legend(loc="upper right")
ax2.hist(results.storage, edgecolor="green", facecolor="none", bins=20, label="storage")
ax2.legend(loc="upper right")

# display the plot
plt.tight_layout()
plt.show()

# %%
