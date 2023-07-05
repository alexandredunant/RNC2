import pandas as pd
import numpy as np

from mods_catchment_flow import *
from mod_river_flow import *

# Define catchments and their corresponding parameters
catchments = pd.read_csv("../../../docs/Catchments_updt_sorted.csv", index_col=0)
catchments = catchments[catchments['River System'] == 'Rangitaki']
catchments['streamflow'] = np.zeros(len(catchments))
catchments['storage'] = np.zeros(len(catchments))
catchments['max_soil_moisture'] = np.random.randint(200, 400, len(catchments))

# Convert df into a dictionary
river_network = {}
water_flow = {}
keys = catchments.Target_river
source_nodes = catchments.Source_river
baseline_river = 0  # baseline value for river flow
catchment_names = catchments['Catchment Name']

# create river network where each edge is a tuple of the form (target_node, flow_rate) and a water flow dict (start
# with a mean value of 0 m3/s)
for i in range(len(keys)):
    river_network[keys[i]] = {'source_river': source_nodes[i], 
                              'edge_flow': baseline_river,
                              'catchment_name': catchment_names[i]}
    water_flow[keys[i]] = baseline_river

# generate a list of precipitation for the time of simulation (in days)
num_days = 365
precipitations = generate_daily_rainfall_amount(num_days)

record = []
df_catchments = catchments.copy()

# Iterate over timesteps
for t in trange(1, num_days):
    # generate daily rainfall amount
    precipitation = precipitations[t]

    # NLRRM model for the catchment - storage and streamflow for each catchment
    df_catchments = simulate_catchment_reservoir_flow(df_catchments, precipitation, dtime=1)

    # Create the total flow being routed from catchments to each river node per day
    water_flow = catchment_flow_to_river_nodes(df_catchments, water_flow=water_flow)

    # simulate river behavior for a number of stepped iteration (hours?) per day
    water_flow = simulate_river_flow(river_network, water_flow, 24, 0.967)

    # update the new river flow state at the end of the day
    for node in river_network:
        if water_flow[node] == 0:
            river_network[node]['edge_flow'] == baseline_river
        else:
            river_network[node]['edge_flow'] = water_flow[node]

    #  Record results
    df = pd.concat([pd.Series(river_network.keys()),
                    pd.Series([river_network[key]['source_river'] for key in river_network]),
                    pd.Series([river_network[key]['edge_flow'] for key in river_network])], axis=1)
    df.columns = ['source', 'target', 'river_flow']
    df['precipitation'] = precipitation
    df['storage'] = df_catchments['storage']
    df['time'] = t

    record.append(df)

    results = pd.concat(record)

print("simulation finished")


res = results[["time", "precipitation", "storage", "river_flow"]].groupby('time').mean().reset_index()




################################################################
# Plot results
################################################################

import matplotlib.pyplot as plt

# create the figure and the axes
# create the figure
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)

# set the x-axis and the first y-axis (precipitation)
ax1.set_xlabel('Time')
ax1.set_ylabel('Precipitation', color='blue')
ax1.step(res.time, res.precipitation, lw=0.5, color='blue')

# create the second y-axis (discharge)
ax2 = ax1.twinx()
ax2.set_ylabel('Discharge', color='red')
ax2.plot(res.time, res.river_flow, lw=0.3, color='red')

# set the y-axis color to green
ax1.tick_params(axis='y', colors='blue')
ax2.tick_params(axis='y', colors='red')

# create the second subplot
ax3 = fig.add_subplot(2, 1, 2)
ax3.step(res.time, res.precipitation, lw=0.5, color='blue')
ax3.set_xlabel('Time')
ax3.set_ylabel('Precipitation', color='blue')

# create the second y-axis (discharge)
ax4 = ax3.twinx()
ax4.set_ylabel('Storage', color='green')
ax4.step(res.time, res.storage, color='green', lw=0.3)

# set the y-axis color to green
ax3.tick_params(axis='y', colors='blue')
ax4.tick_params(axis='y', colors='green')

# set the title of the plot
# plt.title('Precipitation and Discharge vs Time')

# display the plot
plt.tight_layout()
plt.show()

